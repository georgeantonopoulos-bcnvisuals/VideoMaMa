#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install   torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0   --index-url "${TORCH_INDEX_URL}"
python -m pip install   accelerate==1.9.0   diffusers==0.31.0   transformers==4.57.0   tokenizers==0.22.1   "numpy<2"   opencv-python-headless==4.10.0.84   pillow   tqdm   scipy   einops   safetensors   imageio   imageio-ffmpeg   huggingface_hub   psutil   OpenEXR   Imath

mkdir -p "${CHECKPOINT_DIR}"
export VIDEOMAMA_ROOT="${ROOT_DIR}"
python - <<'INNERPY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["VIDEOMAMA_ROOT"])
ckpt = root / "checkpoints"
base = ckpt / "stable-video-diffusion-img2vid-xt"
vm = ckpt / "VideoMaMa"

if not base.exists():
    snapshot_download(
        repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
        local_dir=str(base),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

if not vm.exists():
    snapshot_download(
        repo_id="SammyLim/VideoMaMa",
        local_dir=str(vm),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
INNERPY

echo
printf '[VideoMaMa] setup complete
'
printf 'Activate with: source %s/bin/activate
' "${VENV_DIR}"
printf 'Base model: %s/stable-video-diffusion-img2vid-xt
' "${CHECKPOINT_DIR}"
printf 'VideoMaMa: %s/VideoMaMa
' "${CHECKPOINT_DIR}"
