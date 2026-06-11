#!/usr/bin/env bash
# Launch the SAM 3 + VideoMaMa production sequence UI.
#
# Resolves venv + checkpoint paths from .videomama-env (written by
# bootstrap_tmp_venv.sh). Override any of those env vars in the calling shell
# before invoking this script if you need to point at a different venv or
# checkpoints location.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.videomama-env"
APP_PATH="${REPO_ROOT}/demo/production_frame_app.py"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

UI_VENV="${VIDEOMAMA_UI_VENV:-/tmp/videomama-sam3-ui-venv}"

if [[ ! -d "${UI_VENV}" ]]; then
  echo "Missing SAM 3 UI environment: ${UI_VENV}" >&2
  echo "Bootstrap it first:" >&2
  echo "  bash ${REPO_ROOT}/scripts/bootstrap_tmp_venv.sh sam3-ui" >&2
  echo "The bootstrap script will auto-detect python3.12 from PATH or pyenv." >&2
  exit 1
fi

if [[ ! -f "${APP_PATH}" ]]; then
  echo "Missing app: ${APP_PATH}" >&2
  exit 1
fi

export VIDEOMAMA_UI_HOST="${VIDEOMAMA_UI_HOST:-127.0.0.1}"
export VIDEOMAMA_UI_PORT="${VIDEOMAMA_UI_PORT:-7861}"
export VIDEOMAMA_UI_SHARE="${VIDEOMAMA_UI_SHARE:-1}"
export SAM3_MODEL_VERSION="${SAM3_MODEL_VERSION:-sam3}"

# SAM 3 and VideoMaMa share one GPU, so the allocator churns between two large
# models across many chunks. expandable_segments lets PyTorch grow allocations in
# place instead of fragmenting, which avoids spurious OOMs on long sequences.
# torch >= 2.5 reads PYTORCH_ALLOC_CONF; the older name still works as an alias.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Gradio resolves some cache/temp paths relative to the process working directory
# and calls getcwd() per request. The repo lives on a network mount whose cwd can
# be swapped out mid-session, after which getcwd() fails and Gradio's file routes
# crash. Pin these to absolute local paths and run from a stable cwd.
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-${TMPDIR:-/tmp}/videomama-gradio}"
export GRADIO_EXAMPLES_CACHE="${GRADIO_EXAMPLES_CACHE:-${GRADIO_TEMP_DIR}/examples}"
mkdir -p "${GRADIO_TEMP_DIR}" "${GRADIO_EXAMPLES_CACHE}"
cd "${GRADIO_TEMP_DIR}"

# shellcheck disable=SC1091
source "${UI_VENV}/bin/activate"
if ! python -c "import gradio, fastapi, starlette; assert tuple(map(int, starlette.__version__.split('.')[:2])) < (0, 39)" >/dev/null 2>&1; then
  echo "SAM 3 UI environment is missing compatible Gradio dependencies." >&2
  echo "Refresh it with:" >&2
  echo "  ${UI_VENV}/bin/python -m pip install -r ${REPO_ROOT}/scripts/requirements-sam3-ui.txt" >&2
  exit 1
fi
exec python "${APP_PATH}"
