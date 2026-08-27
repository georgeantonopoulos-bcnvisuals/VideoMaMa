#!/usr/bin/env bash
# Bootstrap VideoMaMa runtime on any Rocky/RHEL 9-class machine with a CUDA GPU.
#
# Builds the runtime venvs under VIDEOMAMA_VENV_ROOT (default
# /mnt/temporal/VideoMama, a large local scratch disk) and wires them to the
# shared checkpoints at CHECKPOINTS_DIR. Code lives wherever you cloned/rsynced
# this repo; nothing is written outside REPO_ROOT except the venv root and
# optional ~/.cache/huggingface.
#
# The venvs are disposable but no longer ephemeral: they survive reboots, so a
# rebuild is only needed when dependencies change. Override the location with
# VIDEOMAMA_VENV_ROOT, or pin an individual venv with VIDEOMAMA_VENV /
# VIDEOMAMA_UI_VENV / VIDEOMAMA_SAM2MATTING_VENV.
#
# Usage:
#   bash scripts/bootstrap_tmp_venv.sh              # both core venvs
#   bash scripts/bootstrap_tmp_venv.sh inference    # inference venv only
#   bash scripts/bootstrap_tmp_venv.sh sam3-ui      # UI venv only
#   bash scripts/bootstrap_tmp_venv.sh sam2matting  # isolated SAM2Matting runtime
#   bash scripts/bootstrap_tmp_venv.sh everything   # all three
#
# The SAM2Matting runtime is intentionally a separate venv + source checkout +
# process. It vendors modified top-level `sam2`/`sam3` packages that would shadow
# the official SAM 3 install if they shared an environment. See
# scripts/bootstrap_sam2matting.sh.
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
HF_TOKEN_FILE="${VIDEOMAMA_HF_TOKEN_FILE:-${REPO_ROOT}/.videomama-secrets/huggingface.token}"

DEFAULT_CHECKPOINTS="/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints"
CHECKPOINTS_DIR="${VIDEOMAMA_CHECKPOINTS:-${DEFAULT_CHECKPOINTS}}"

# One knob for where every runtime lives. /tmp on these hosts is the root
# filesystem and runs out of space long before three CUDA venvs fit.
VENV_ROOT="${VIDEOMAMA_VENV_ROOT:-/mnt/temporal/VideoMama}"

INFER_VENV="${VIDEOMAMA_VENV:-${VENV_ROOT}/videomama-venv}"
UI_VENV="${VIDEOMAMA_UI_VENV:-${VENV_ROOT}/videomama-sam3-ui-venv}"
S2M_VENV="${VIDEOMAMA_SAM2MATTING_VENV:-${VENV_ROOT}/videomama-sam2matting-venv}"
S2M_HOME="${VIDEOMAMA_SAM2MATTING_HOME:-${VENV_ROOT}/videomama-sam2matting-src}"

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

  if [[ -d "${INFER_VENV}" && ! -x "${INFER_VENV}/bin/python" ]]; then
    log "Removing incomplete inference venv: ${INFER_VENV}"
    rm -rf "${INFER_VENV}"
  fi
  if [[ ! -x "${INFER_VENV}/bin/python" ]]; then
    "${PYTHON_INFER_BIN}" -m venv "${INFER_VENV}"
  fi
  # shellcheck disable=SC1091
  source "${INFER_VENV}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "${INFER_TORCH_INDEX_URL}" \
      torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  python -m pip install -r "${REPO_ROOT}/scripts/requirements-inference.txt"
  mkdir -p "${REPO_ROOT}/tmp/runtime-locks"
  python -m pip freeze > "${REPO_ROOT}/tmp/runtime-locks/requirements-inference.lock.txt"
  deactivate
  log "Inference venv ready."
}

build_ui_venv() {
  PYTHON_UI_BIN="$(resolve_ui_python)"
  log "Building SAM 3 UI venv at ${UI_VENV} (Python: ${PYTHON_UI_BIN})"
  check_python "${PYTHON_UI_BIN}" "ui"

  if [[ -d "${UI_VENV}" && ! -x "${UI_VENV}/bin/python" ]]; then
    log "Removing incomplete SAM 3 UI venv: ${UI_VENV}"
    rm -rf "${UI_VENV}"
  fi
  if [[ ! -x "${UI_VENV}/bin/python" ]]; then
    "${PYTHON_UI_BIN}" -m venv "${UI_VENV}"
  fi
  # shellcheck disable=SC1091
  source "${UI_VENV}/bin/activate"
  # requirements-sam3-ui.txt intentionally pins setuptools<82 for SAM 3.
  # Keep the generic bootstrap upgrade from fighting that pin on every run.
  python -m pip install --upgrade pip wheel
  python -m pip install --index-url "${UI_TORCH_INDEX_URL}" \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
  python -m pip install -r "${REPO_ROOT}/scripts/requirements-sam3-ui.txt"
  mkdir -p "${REPO_ROOT}/tmp/runtime-locks"
  local sam3_lock="${REPO_ROOT}/tmp/runtime-locks/sam3-requirement.txt"
  if [[ -s "${sam3_lock}" && "${REFRESH_SAM3_LOCK:-0}" != "1" ]]; then
    log "Installing the previously resolved SAM 3 commit from ${sam3_lock}"
    python -m pip install -r "${sam3_lock}"
  else
    local sam3_ref="${SAM3_GIT_REF:-main}"
    log "Installing SAM 3 from Git ref ${sam3_ref}"
    python -m pip install "git+https://github.com/facebookresearch/sam3.git@${sam3_ref}"
    python -m pip freeze | awk '/^sam3 @ git\+https:\/\/github.com\/facebookresearch\/sam3.git@/ { print; exit }' > "${sam3_lock}"
    if [[ ! -s "${sam3_lock}" ]]; then
      log "WARNING: could not record the resolved SAM 3 Git commit; the full environment lock is still available."
    fi
  fi
  python -m pip freeze > "${REPO_ROOT}/tmp/runtime-locks/requirements-sam3-ui.lock.txt"
  deactivate
  log "SAM 3 UI venv ready."
}

build_sam2matting_runtime() {
  log "Delegating to scripts/bootstrap_sam2matting.sh (isolated runtime)"
  VIDEOMAMA_CHECKPOINTS="${CHECKPOINTS_DIR}" \
  VIDEOMAMA_VENV_ROOT="${VENV_ROOT}" \
  VIDEOMAMA_SAM2MATTING_VENV="${S2M_VENV}" \
  VIDEOMAMA_SAM2MATTING_HOME="${S2M_HOME}" \
  bash "${REPO_ROOT}/scripts/bootstrap_sam2matting.sh" all
}

write_env_file() {
  log "Writing env file: ${ENV_FILE}"
  # The env file is rewritten whole on every run, including single-target runs.
  # Resolve the UI interpreter even when this run did not build the UI venv, so
  # `bootstrap_tmp_venv.sh sam2matting` cannot blank a value the UI build set.
  if [[ -z "${PYTHON_UI_BIN}" ]]; then
    PYTHON_UI_BIN="$(resolve_ui_python)"
  fi
  # First reachable ACES 1.2 config wins; blank falls through to the app's own
  # search order, which ends at OCIO's built-in ACES config.
  ACES_OCIO_CONFIG="${VIDEOMAMA_OCIO_CONFIG:-}"
  if [[ -z "${ACES_OCIO_CONFIG}" ]]; then
    for candidate in \
      "/mnt/production/user/george.antonopoulos/DEV/aces_1.2/config.ocio" \
      "/mnt/production/project/bcn_lib/work/config/aces_1.2/config.ocio"; do
      if [[ -f "${candidate}" ]]; then
        ACES_OCIO_CONFIG="${candidate}"
        break
      fi
    done
  fi
  cat > "${ENV_FILE}" <<EOF
# Generated by scripts/bootstrap_tmp_venv.sh on $(date -Iseconds)
export VIDEOMAMA_ROOT="${REPO_ROOT}"
export VIDEOMAMA_CHECKPOINTS="${CHECKPOINTS_DIR}"
export VIDEOMAMA_VENV_ROOT="${VENV_ROOT}"
export VIDEOMAMA_VENV="${INFER_VENV}"
export VIDEOMAMA_UI_VENV="${UI_VENV}"
export VIDEOMAMA_UI_PYTHON="${PYTHON_UI_BIN}"
export VIDEOMAMA_HF_TOKEN_FILE="${HF_TOKEN_FILE}"
# Share only the credential file. Hugging Face model caches remain local to
# each machine, and an explicitly supplied HF_TOKEN_PATH still takes priority.
export HF_TOKEN_PATH="\${HF_TOKEN_PATH:-\${VIDEOMAMA_HF_TOKEN_FILE}}"
export VIDEOMAMA_SAM2MATTING_VENV="${S2M_VENV}"
export VIDEOMAMA_SAM2MATTING_HOME="${S2M_HOME}"
export VIDEOMAMA_BASE_MODEL_PATH="\${VIDEOMAMA_BASE_MODEL_PATH:-${CHECKPOINTS_DIR}/stable-video-diffusion-img2vid-xt}"
export VIDEOMAMA_UNET_CHECKPOINT_PATH="\${VIDEOMAMA_UNET_CHECKPOINT_PATH:-${CHECKPOINTS_DIR}/VideoMaMa}"
# EXR plates are ACES scene-linear. The UI resolves this config to apply the
# ACES view transform; leave it unset to use the app's built-in search order.
export VIDEOMAMA_OCIO_CONFIG="\${VIDEOMAMA_OCIO_CONFIG:-${ACES_OCIO_CONFIG}}"
EOF
}

if ! mkdir -p "${VENV_ROOT}" 2>/dev/null || [[ ! -w "${VENV_ROOT}" ]]; then
  echo "Venv root is not writable: ${VENV_ROOT}" >&2
  echo "Create it, or point VIDEOMAMA_VENV_ROOT somewhere with tens of GB free." >&2
  exit 1
fi
log "Venv root: ${VENV_ROOT} ($(df -h --output=avail "${VENV_ROOT}" 2>/dev/null | tail -1 | tr -d ' ') available)"

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
  sam2matting)
    build_sam2matting_runtime
    ;;
  everything)
    build_inference_venv
    build_ui_venv
    build_sam2matting_runtime
    ;;
  *)
    echo "Unknown target: ${target} (expected: all | inference | sam3-ui | sam2matting | everything)" >&2
    exit 2
    ;;
esac

write_env_file

cat <<MSG

[videomama-bootstrap] Done.

Activate inference env:
    source "${ENV_FILE}"
    source "\${VIDEOMAMA_VENV}/bin/activate"

Build the isolated SAM2Matting runtime (separate venv/checkout/process):
    bash "${REPO_ROOT}/scripts/bootstrap_sam2matting.sh"

Launch SAM 3 production UI:
    bash "${REPO_ROOT}/scripts/run_production_frame_ui.sh"

Venv root:   ${VENV_ROOT}
Checkpoints: ${CHECKPOINTS_DIR}
MSG
