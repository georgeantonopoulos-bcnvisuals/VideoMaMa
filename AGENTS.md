# AGENTS.md

## Project Overview

This repository is a working fork of VideoMaMa, "Mask-Guided Video Matting via Generative Prior." It contains the upstream training and inference code plus a custom production UI/harness that combines SAM2 mask tracking with VideoMaMa matting over image or EXR sequences.

Core pieces:

- `pipeline_svd_mask.py`: VideoMaMa inference pipeline built around Stable Video Diffusion with mask conditioning.
- `inference_onestep_folder.py`: batch folder inference entry point for pre-existing image and mask sequence folders.
- `demo/production_frame_app.py`: custom Gradio production UI for loading a sequence, adding SAM2 keyframe prompts, generating masks, and running VideoMaMa over the shot.
- `demo/sam2_wrapper_hf.py`: production SAM2 wrapper used by `production_frame_app.py`; supports multi-keyframe prompting and forward/reverse propagation.
- `demo/videomama_wrapper.py`: VideoMaMa wrapper used by the production app.
- `scripts/bootstrap_tmp_venv.sh`: builds local runtime venvs in `/tmp` and writes `.videomama-env`.
- `scripts/run_production_frame_ui.sh`: launches the production Gradio UI.

## Runtime Layout

Production machines keep code, weights, and venvs separate:

- Code/config: this repo, usually `/mnt/production/user/<username>/DEV/AI/VideoMama/`.
- Shared checkpoints: `/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints/`.
- Inference venv: `/tmp/videomama-venv/` using Python 3.9.
- SAM2 UI venv: `/tmp/videomama-sam2-ui-venv/` using Python 3.10.

The upstream `sam2` package does not support Python 3.9, so the UI and base inference environments are intentionally split.

Important environment variables are written to `.videomama-env`:

- `VIDEOMAMA_ROOT`
- `VIDEOMAMA_CHECKPOINTS`
- `VIDEOMAMA_VENV`
- `VIDEOMAMA_UI_VENV`
- `SAM2_CHECKPOINT_PATH`
- `VIDEOMAMA_BASE_MODEL_PATH`
- `VIDEOMAMA_UNET_CHECKPOINT_PATH`

Do not hardcode new checkpoint paths if these env vars will work.

## Setup And Launch

Bootstrap both local venvs:

```bash
bash scripts/bootstrap_tmp_venv.sh
source .videomama-env
```

Bootstrap only one environment:

```bash
bash scripts/bootstrap_tmp_venv.sh inference
bash scripts/bootstrap_tmp_venv.sh sam2-ui
```

Launch the production SAM2 + VideoMaMa UI:

```bash
bash scripts/run_production_frame_ui.sh
```

Defaults:

- Host: `127.0.0.1`
- Port: `7861`
- Gradio share: enabled unless `VIDEOMAMA_UI_SHARE=0`

Override at launch with environment variables, for example:

```bash
VIDEOMAMA_UI_PORT=7862 VIDEOMAMA_UI_SHARE=0 bash scripts/run_production_frame_ui.sh
```

## Production UI Flow

`demo/production_frame_app.py` does the following:

1. Loads an image or EXR sequence from a directory.
2. Converts frames to RGB and caches resized 1024x576 JPEGs under `tmp/production_sequence_app/<timestamp>_<sequence>/sam2_frames`.
3. Lets the user add positive/negative SAM2 point prompts on multiple keyframes.
4. Saves keyframe prompts to `keyframe_prompts.json`.
5. Uses `sam2_wrapper_hf.SAM2VideoTracker.track_video_from_dir()` to propagate masks across the cached sequence.
6. Saves masks under `sam2_masks`.
7. Runs VideoMaMa in chunks with configurable overlap.
8. Saves outputs under `videomama_frames` and grayscale alpha previews under `alpha_frames`.

The production app supports `.exr`, `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` inputs. EXR RGB conversion uses `OpenEXR`/`Imath`, `exr_gamma`, and exposure handling in code.

## Inference CLI

Use `inference_onestep_folder.py` when masks already exist:

```bash
python inference_onestep_folder.py \
  --base_model_path "$VIDEOMAMA_BASE_MODEL_PATH" \
  --unet_checkpoint_path "$VIDEOMAMA_UNET_CHECKPOINT_PATH" \
  --image_root_path /path/to/image \
  --mask_root_path /path/to/mask \
  --output_dir /path/to/output \
  --keep_aspect_ratio
```

Expected folder shape:

```text
image/<sequence_name>/<frame>.png|exr|...
mask/<sequence_name>/<matching_frame>.png|exr|...
```

The CLI supports EXR input and mask channel selection. See `inference.md` for flags such as `--mask_channel`, `--exr_gamma`, and `--exr_exposure`.

## Dependency Notes

- `scripts/requirements-inference.txt` is the Python 3.9 inference dependency set.
- `scripts/requirements-sam2-ui.txt` is the Python 3.10 production UI dependency set and installs SAM2 from GitHub.
- Torch is installed separately by `scripts/bootstrap_tmp_venv.sh` from the CUDA 12.4 wheel index.
- `setup.py` contains the broader upstream/training dependency list and includes heavy training dependencies such as `deepspeed` and `flash_attn`.

Avoid mixing the UI and inference venv assumptions unless you are explicitly changing the runtime layout.

## Training And Data

- `train.py` is the main training entry point.
- `training_scripts/train_stage1.sh` trains spatial layers on single frames.
- `training_scripts/train_stage2.sh` resumes from stage 1 and trains temporal layers on video clips.
- `data_pipeline/generate_synthetic.py` and `data_pipeline/config/fg_alpha_bg_paths.yaml` handle synthetic data generation.
- Training docs assume S3-capable workflows, but local storage can be used if paths are adjusted.

## Development Guidance

- The worktree may contain existing local changes; do not revert them unless explicitly asked.
- Prefer editing only the custom production harness files when fixing SAM2 + VideoMaMa UI behavior:
  - `demo/production_frame_app.py`
  - `demo/sam2_wrapper_hf.py`
  - `demo/videomama_wrapper.py`
  - `scripts/bootstrap_tmp_venv.sh`
  - `scripts/run_production_frame_ui.sh`
  - `scripts/requirements-*.txt`
- Be careful with `demo/sam2_wrapper.py`: it is the older wrapper and contains a hardcoded external SAM2 path. The production app imports `sam2_wrapper_hf.py`.
- Generated app outputs belong under `tmp/production_sequence_app/` and should not be treated as source.
- Checkpoint directories can be very large; do not vendor weights into the repo.
- Preserve the 1024x576 working size unless you also audit VideoMaMa assumptions in the wrappers and pipeline.

## Useful Checks

Basic import/syntax checks without starting a GPU workload:

```bash
python -m py_compile demo/production_frame_app.py demo/sam2_wrapper_hf.py demo/videomama_wrapper.py
```

Check environment wiring:

```bash
source .videomama-env
printf '%s\n' "$VIDEOMAMA_CHECKPOINTS" "$SAM2_CHECKPOINT_PATH" "$VIDEOMAMA_BASE_MODEL_PATH" "$VIDEOMAMA_UNET_CHECKPOINT_PATH"
```

Starting the UI or running inference can require GPU access and large checkpoints. Do not assume those are available in every coding session.
