#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_VENV="${ROOT_DIR}/.venv-sam2-ui"
APP_PATH="${ROOT_DIR}/demo/production_frame_app.py"

if [[ ! -d "${UI_VENV}" ]]; then
  echo "Missing UI environment: ${UI_VENV}" >&2
  echo "Build the SAM2 UI environment first." >&2
  exit 1
fi

if [[ ! -f "${APP_PATH}" ]]; then
  echo "Missing app: ${APP_PATH}" >&2
  exit 1
fi

export VIDEOMAMA_UI_HOST="${VIDEOMAMA_UI_HOST:-127.0.0.1}"
export VIDEOMAMA_UI_PORT="${VIDEOMAMA_UI_PORT:-7861}"
export VIDEOMAMA_UI_SHARE="${VIDEOMAMA_UI_SHARE:-1}"
export SAM2_CHECKPOINT_PATH="${SAM2_CHECKPOINT_PATH:-${ROOT_DIR}/checkpoints/sam2.1_hiera_large.pt}"

source "${UI_VENV}/bin/activate"
exec python "${APP_PATH}"
