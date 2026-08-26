#!/usr/bin/env bash
# Build enough local runtime from a bare user account to launch the
# VideoMaMa SAM 3 production Gradio UI.
#
# This script is intentionally user-space first:
# - installs pyenv under ~/.pyenv when missing
# - builds local libffi and bzip2 so Python 3.12 has ctypes and bz2 even when
#   system *-devel RPMs are unavailable
# - builds the SAM 3 UI venv through bootstrap_tmp_venv.sh
# - reuses the shared project Hugging Face token, or HF_TOKEN when provided
# - launches the Gradio UI
#
# Usage:
#   bash scripts/bootstrap_from_scratch_and_run_ui.sh
#
# Useful overrides:
#   PYTHON_UI_VERSION=3.12.13
#   VIDEOMAMA_UI_PORT=7864
#   VIDEOMAMA_UI_SHARE=0
#   HF_TOKEN=hf_...                         # optional, for gated SAM 3 access
#   ALLOW_NO_HF_AUTH=1                      # launch UI even if SAM 3 auth is missing
#   REQUIRE_CUDA=0                          # skip CUDA preflight for non-GPU testing
#   SKIP_LAUNCH=1                           # build only
#   INSTALL_INFERENCE=1                     # also build ${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-venv

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [[ -z "${SCRIPT_DIR}" ]]; then
  echo "Cannot resolve this script's directory. cd into the repo and re-run." >&2
  exit 1
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Reuse the project-scoped credential from the shared checkout. This does not
# share Hugging Face caches or put the token itself in the environment file.
export VIDEOMAMA_HF_TOKEN_FILE="${VIDEOMAMA_HF_TOKEN_FILE:-${REPO_ROOT}/.videomama-secrets/huggingface.token}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${VIDEOMAMA_HF_TOKEN_FILE}}"

if [[ -n "${SUDO_USER:-}" ]]; then
  echo "Do not run this script with sudo. It installs user-local pyenv/deps and /tmp venvs." >&2
  exit 1
fi

PYTHON_UI_VERSION="${PYTHON_UI_VERSION:-3.12.13}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
PYENV_BIN="${PYENV_ROOT}/bin/pyenv"
LOCAL_ROOT="${LOCAL_ROOT:-${HOME}/.local}"
LOCAL_SRC="${LOCAL_ROOT}/src"
LIBFFI_VERSION="${LIBFFI_VERSION:-3.4.6}"
BZIP2_VERSION="${BZIP2_VERSION:-1.0.8}"
LIBFFI_PREFIX="${LIBFFI_PREFIX:-${LOCAL_ROOT}/libffi-${LIBFFI_VERSION}}"
BZIP2_PREFIX="${BZIP2_PREFIX:-${LOCAL_ROOT}/bzip2-${BZIP2_VERSION}}"

log() { printf '[videomama-from-scratch] %s\n' "$*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

maybe_install_system_packages() {
  if ! need_cmd dnf || ! need_cmd sudo; then
    return 0
  fi
  if ! sudo -n true >/dev/null 2>&1; then
    log "Passwordless sudo is unavailable; using local lib builds for Python headers/libs."
    return 0
  fi

  log "Installing system build prerequisites with dnf."
  sudo dnf install -y \
    git curl make gcc gcc-c++ patch tar gzip xz \
    openssl-devel zlib-devel xz-devel \
    libffi-devel bzip2-devel ncurses-devel readline-devel sqlite-devel tk-devel
}

install_pyenv() {
  if [[ -x "${PYENV_BIN}" ]]; then
    log "pyenv already present: ${PYENV_BIN}"
    return 0
  fi
  if ! need_cmd git; then
    echo "git is required to install pyenv. Install git or make it available on PATH." >&2
    exit 1
  fi
  log "Installing pyenv into ${PYENV_ROOT}"
  mkdir -p "$(dirname "${PYENV_ROOT}")"
  git clone https://github.com/pyenv/pyenv.git "${PYENV_ROOT}"
}

build_local_libffi() {
  if [[ -f "${LIBFFI_PREFIX}/include/ffi.h" && -f "${LIBFFI_PREFIX}/lib64/libffi.so" ]]; then
    log "local libffi already present: ${LIBFFI_PREFIX}"
    return 0
  fi
  log "Building local libffi ${LIBFFI_VERSION}"
  mkdir -p "${LOCAL_SRC}"
  cd "${LOCAL_SRC}"
  curl -L "https://github.com/libffi/libffi/releases/download/v${LIBFFI_VERSION}/libffi-${LIBFFI_VERSION}.tar.gz" \
    -o "libffi-${LIBFFI_VERSION}.tar.gz"
  rm -rf "libffi-${LIBFFI_VERSION}"
  tar -xzf "libffi-${LIBFFI_VERSION}.tar.gz"
  cd "libffi-${LIBFFI_VERSION}"
  ./configure --prefix="${LIBFFI_PREFIX}"
  make -j"$(nproc 2>/dev/null || echo 4)"
  make install
  cd "${REPO_ROOT}"
}

build_local_bzip2() {
  if [[ -f "${BZIP2_PREFIX}/include/bzlib.h" && -f "${BZIP2_PREFIX}/lib/libbz2.so" ]]; then
    log "local bzip2 already present: ${BZIP2_PREFIX}"
    return 0
  fi
  log "Building local bzip2 ${BZIP2_VERSION}"
  mkdir -p "${LOCAL_SRC}"
  cd "${LOCAL_SRC}"
  curl -L "https://sourceware.org/pub/bzip2/bzip2-${BZIP2_VERSION}.tar.gz" \
    -o "bzip2-${BZIP2_VERSION}.tar.gz"
  rm -rf "bzip2-${BZIP2_VERSION}"
  tar -xzf "bzip2-${BZIP2_VERSION}.tar.gz"
  cd "bzip2-${BZIP2_VERSION}"
  make clean || true
  make CFLAGS="-fPIC -Wall -Winline -O2 -g -D_FILE_OFFSET_BITS=64" -j"$(nproc 2>/dev/null || echo 4)"
  make -f Makefile-libbz2_so clean || true
  make -f Makefile-libbz2_so CFLAGS="-fPIC -Wall -Winline -O2 -g -D_FILE_OFFSET_BITS=64"
  make install PREFIX="${BZIP2_PREFIX}"
  cp -f "libbz2.so.${BZIP2_VERSION}" "${BZIP2_PREFIX}/lib/"
  ln -sf "libbz2.so.${BZIP2_VERSION}" "${BZIP2_PREFIX}/lib/libbz2.so.1.0"
  ln -sf "libbz2.so.${BZIP2_VERSION}" "${BZIP2_PREFIX}/lib/libbz2.so"
  cd "${REPO_ROOT}"
}

install_python_ui() {
  export PYENV_ROOT
  export PATH="${PYENV_ROOT}/bin:${PATH}"

  if "${PYENV_BIN}" versions --bare | grep -qx "${PYTHON_UI_VERSION}"; then
    if "${PYENV_ROOT}/versions/${PYTHON_UI_VERSION}/bin/python" -c "import bz2, ctypes, ssl, zlib, venv" >/dev/null 2>&1; then
      log "pyenv Python ${PYTHON_UI_VERSION} already has required modules."
      return 0
    fi
    log "Existing Python ${PYTHON_UI_VERSION} is incomplete; rebuilding it."
    "${PYENV_BIN}" uninstall -f "${PYTHON_UI_VERSION}"
  fi

  build_local_libffi
  build_local_bzip2

  export CPPFLAGS="-I${LIBFFI_PREFIX}/include -I${BZIP2_PREFIX}/include ${CPPFLAGS:-}"
  export LDFLAGS="-L${LIBFFI_PREFIX}/lib64 -Wl,-rpath,${LIBFFI_PREFIX}/lib64 -L${BZIP2_PREFIX}/lib -Wl,-rpath,${BZIP2_PREFIX}/lib ${LDFLAGS:-}"
  export PKG_CONFIG_PATH="${LIBFFI_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

  log "Installing pyenv Python ${PYTHON_UI_VERSION}"
  "${PYENV_BIN}" install "${PYTHON_UI_VERSION}"
  "${PYENV_ROOT}/versions/${PYTHON_UI_VERSION}/bin/python" -c "import bz2, ctypes, ssl, zlib, venv"
}

build_venvs() {
  local target="sam3-ui"
  if [[ "${INSTALL_INFERENCE:-0}" == "1" ]]; then
    target="all"
  fi

  export PYTHON_UI_BIN="${PYENV_ROOT}/versions/${PYTHON_UI_VERSION}/bin/python"
  export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

  log "Building VideoMaMa runtime target: ${target}"
  bash "${REPO_ROOT}/scripts/bootstrap_tmp_venv.sh" "${target}"
}

login_huggingface_if_requested() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN is not set; using any existing Hugging Face CLI login."
    return 0
  fi

  local hf_bin="${VIDEOMAMA_UI_VENV:-${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-sam3-ui-venv}/bin/hf"
  if [[ ! -x "${hf_bin}" ]]; then
    echo "Cannot find Hugging Face CLI at ${hf_bin}" >&2
    exit 1
  fi

  log "Logging into Hugging Face with HF_TOKEN."
  "${hf_bin}" auth login --token "${HF_TOKEN}"
}

verify_ui_environment() {
  local py="${VIDEOMAMA_UI_VENV:-${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-sam3-ui-venv}/bin/python"
  log "Verifying UI Python imports."
  "${py}" -c "import bz2, ctypes, torch, torchvision, gradio, cv2, OpenEXR, sam3; print('ui_imports_ok')"

  "${py}" -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
}

verify_cuda_available() {
  if [[ "${REQUIRE_CUDA:-1}" != "1" ]]; then
    log "REQUIRE_CUDA is disabled; skipping CUDA preflight."
    return 0
  fi

  local py="${VIDEOMAMA_UI_VENV:-${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-sam3-ui-venv}/bin/python"
  log "Checking CUDA visibility for SAM 3."
  "${py}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)" || {
    echo "CUDA is not visible to PyTorch in ${py}; SAM 3 model initialization cannot work." >&2
    exit 1
  }
}

verify_huggingface_sam3_access() {
  if [[ "${ALLOW_NO_HF_AUTH:-0}" == "1" ]]; then
    log "ALLOW_NO_HF_AUTH=1; skipping SAM 3 gated-repo access check."
    return 0
  fi

  local hf_bin="${VIDEOMAMA_UI_VENV:-${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-sam3-ui-venv}/bin/hf"
  local py="${VIDEOMAMA_UI_VENV:-${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}/videomama-sam3-ui-venv}/bin/python"
  local model_version="${SAM3_MODEL_VERSION:-sam3}"
  local repo_id="facebook/sam3"
  if [[ "${model_version}" == "sam3.1" ]]; then
    repo_id="facebook/sam3.1"
  fi

  log "Checking Hugging Face auth and gated access for ${repo_id}."
  local whoami
  whoami="$("${hf_bin}" auth whoami 2>&1 || true)"
  if [[ "${whoami}" == *"Not logged in"* || -z "${whoami}" ]]; then
    cat >&2 <<MSG
Hugging Face is not logged in for this user.
SAM 3 model initialization requires access to the gated ${repo_id} repo.

Authenticate before launching:
    ${hf_bin} auth login

Or run with HF_TOKEN=... for non-interactive login.
Set ALLOW_NO_HF_AUTH=1 only if you intentionally want a UI-only launch without model initialization.
MSG
    exit 1
  fi

  "${py}" - "${repo_id}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
hf_hub_download(repo_id=repo_id, filename="config.json")
print(f"sam3_access_ok {repo_id}")
PY
}

launch_ui() {
  if [[ "${SKIP_LAUNCH:-0}" == "1" ]]; then
    log "SKIP_LAUNCH=1; build complete without launching Gradio."
    return 0
  fi

  log "Launching Gradio UI."
  exec bash "${REPO_ROOT}/scripts/run_production_frame_ui.sh"
}

maybe_install_system_packages
install_pyenv
install_python_ui
build_venvs
# shellcheck disable=SC1091
source "${REPO_ROOT}/.videomama-env"
login_huggingface_if_requested
verify_ui_environment
verify_cuda_available
verify_huggingface_sam3_access
launch_ui
