#!/usr/bin/env bash
# Bootstrap VideoMaMa runtime on any Rocky/RHEL 9-class machine with a CUDA GPU.
#
# Builds two ephemeral venvs under /tmp (wiped on reboot on most setups) and
# wires them to the shared checkpoints at CHECKPOINTS_DIR. Code lives wherever
# you cloned/rsynced this repo; nothing is written outside REPO_ROOT except
# /tmp/<venv> and optional ~/.cache/huggingface.
#
# Usage:
#   bash scripts/bootstrap_tmp_venv.sh              # both venvs
#   bash scripts/bootstrap_tmp_venv.sh inference    # inference venv only
#   bash scripts/bootstrap_tmp_venv.sh sam3-ui      # UI venv only
#
# After it finishes, source the generated env file:
#   source "$REPO_ROOT/.videomama-env"

set -euo pipefail

# If the invoking shell's working directory was deleted or swapped out from
# under it (common on network mounts that get remounted or rsynced), getcwd()
# fails and every child process we spawn (pyenv, python, pip) breaks with
# "getcwd: cannot access parent directories". Move into this script's own
# directory first; cd-ing to an absolute path works regardless of whether the
# old cwd is still valid, which repairs the cwd for all subprocesses.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [[ -z "${SCRIPT_DIR}" ]]; then
  echo "Cannot resolve this script's directory." >&2
  echo "Your current directory may have been deleted; cd into a valid directory (e.g. the repo root) and re-run." >&2
  exit 1
fi
cd "${SCRIPT_DIR}"

# Do not run under sudo. sudo switches to root's HOME/PATH, so the user's
# pyenv (~/.pyenv) and python3.12 become invisible and interpreter detection
# fails with a misleading "Missing interpreter". It would also leave the /tmp
# venvs and .videomama-env owned by root, which the normal user cannot use.
# These venvs live in /tmp and the repo, both writable by the invoking user.
if [[ -n "${SUDO_USER:-}" ]]; then
  echo "Do not run this script with sudo." >&2
  echo "It only writes to /tmp and the repo, which your user already owns; sudo hides your" >&2
  echo "pyenv/python3.12 and creates root-owned venvs. Re-run as yourself:" >&2
  echo "    bash ${BASH_SOURCE[0]} ${*:-}" >&2
  exit 1
fi

REPO_ROOT="$(cd .. && pwd)"
ENV_FILE="${REPO_ROOT}/.videomama-env"

DEFAULT_CHECKPOINTS="/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints"
CHECKPOINTS_DIR="${VIDEOMAMA_CHECKPOINTS:-${DEFAULT_CHECKPOINTS}}"

INFER_VENV="${VIDEOMAMA_VENV:-/tmp/videomama-venv}"
UI_VENV="${VIDEOMAMA_UI_VENV:-/tmp/videomama-sam3-ui-venv}"

INFER_TORCH_INDEX_URL="${INFER_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
UI_TORCH_INDEX_URL="${UI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_INFER_BIN="${PYTHON_INFER_BIN:-python3.9}"
PYTHON_UI_BIN="${PYTHON_UI_BIN:-}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
PYENV_BIN="${PYENV_BIN:-${PYENV_ROOT}/bin/pyenv}"

target="${1:-all}"

log() { printf '[videomama-bootstrap] %s\n' "$*"; }

resolve_pyenv_python() {
  local version_prefix="$1"
  local pyenv_cmd="" candidate

  # Standard pyenv installs define `pyenv` as a shell function, so `command -v
  # pyenv` can return the bare name "pyenv" rather than an executable path. That
  # name is not callable from this non-interactive script (its `command pyenv`
  # body needs ~/.pyenv/bin on PATH, which we cannot assume), so only trust
  # command -v when it yields a real executable path. Otherwise fall back to the
  # known pyenv binary location.
  candidate="$(command -v pyenv 2>/dev/null || true)"
  if [[ "${candidate}" == /* && -x "${candidate}" ]]; then
    pyenv_cmd="${candidate}"
  elif [[ -x "${PYENV_BIN}" ]]; then
    pyenv_cmd="${PYENV_BIN}"
  else
    return 1
  fi

  local version
  version="$("${pyenv_cmd}" versions --bare 2>/dev/null | awk -v prefix="${version_prefix}" '$0 ~ "^" prefix "(\\.|$)" { print; exit }')"
  if [[ -z "${version}" ]]; then
    return 1
  fi

  # Prefer pyenv's own resolution, but fall back to the on-disk layout in case
  # `pyenv which` is unhappy in a stripped-down environment.
  local py
  py="$(PYENV_VERSION="${version}" "${pyenv_cmd}" which python 2>/dev/null || true)"
  if [[ -z "${py}" || ! -x "${py}" ]]; then
    py="${PYENV_ROOT}/versions/${version}/bin/python"
  fi
  [[ -x "${py}" ]] || return 1
  printf '%s\n' "${py}"
}

resolve_ui_python() {
  if [[ -n "${PYTHON_UI_BIN}" ]]; then
    printf '%s\n' "${PYTHON_UI_BIN}"
    return 0
  fi

  if resolved="$(resolve_pyenv_python "3.12")"; then
    printf '%s\n' "${resolved}"
    return 0
  fi

  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return 0
  fi

  printf '%s\n' "python3.12"
}

check_python() {
  local bin="$1" label="$2"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "Missing interpreter for $label: $bin" >&2
    echo "Install it (e.g. via pyenv / dnf / uv) or override with PYTHON_${label^^}_BIN=..." >&2
    exit 1
  fi
}

build_inference_venv() {
  log "Building inference venv at ${INFER_VENV} (Python: ${PYTHON_INFER_BIN})"
  check_python "${PYTHON_INFER_BIN}" "infer"

  if [[ ! -d "${INFER_VENV}" ]]; then
    "${PYTHON_INFER_BIN}" -m venv "${INFER_VENV}"
  fi
  # shellcheck disable=SC1091
  source "${INFER_VENV}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${INFER_TORCH_INDEX_URL}" \
      torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  python -m pip install -r "${REPO_ROOT}/scripts/requirements-inference.txt"
  deactivate
  log "Inference venv ready."
}

build_ui_venv() {
  PYTHON_UI_BIN="$(resolve_ui_python)"
  log "Building SAM 3 UI venv at ${UI_VENV} (Python: ${PYTHON_UI_BIN})"
  check_python "${PYTHON_UI_BIN}" "ui"

  if [[ ! -d "${UI_VENV}" ]]; then
    "${PYTHON_UI_BIN}" -m venv "${UI_VENV}"
  fi
  # shellcheck disable=SC1091
  source "${UI_VENV}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${UI_TORCH_INDEX_URL}" \
      torch==2.10.0 torchvision torchaudio
  python -m pip install -r "${REPO_ROOT}/scripts/requirements-sam3-ui.txt"
  deactivate
  log "SAM 3 UI venv ready."
}

write_env_file() {
  log "Writing env file: ${ENV_FILE}"
  cat > "${ENV_FILE}" <<EOF
# Generated by scripts/bootstrap_tmp_venv.sh on $(date -Iseconds)
export VIDEOMAMA_ROOT="${REPO_ROOT}"
export VIDEOMAMA_CHECKPOINTS="${CHECKPOINTS_DIR}"
export VIDEOMAMA_VENV="${INFER_VENV}"
export VIDEOMAMA_UI_VENV="${UI_VENV}"
export VIDEOMAMA_UI_PYTHON="${PYTHON_UI_BIN}"
export VIDEOMAMA_BASE_MODEL_PATH="\${VIDEOMAMA_BASE_MODEL_PATH:-${CHECKPOINTS_DIR}/stable-video-diffusion-img2vid-xt}"
export VIDEOMAMA_UNET_CHECKPOINT_PATH="\${VIDEOMAMA_UNET_CHECKPOINT_PATH:-${CHECKPOINTS_DIR}/VideoMaMa}"
EOF
}

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  log "WARNING: checkpoints dir does not exist yet: ${CHECKPOINTS_DIR}"
  log "         The env file will still be written; populate the directory before running inference."
fi

case "${target}" in
  all)
    build_inference_venv
    build_ui_venv
    ;;
  inference)
    build_inference_venv
    ;;
  sam3-ui)
    build_ui_venv
    ;;
  *)
    echo "Unknown target: ${target} (expected: all | inference | sam3-ui)" >&2
    exit 2
    ;;
esac

write_env_file

cat <<MSG

[videomama-bootstrap] Done.

Activate inference env:
    source "${ENV_FILE}"
    source "\${VIDEOMAMA_VENV}/bin/activate"

Launch SAM 3 production UI:
    bash "${REPO_ROOT}/scripts/run_production_frame_ui.sh"

Checkpoints: ${CHECKPOINTS_DIR}
MSG
