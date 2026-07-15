# AGENTS.md

## Project Overview

This repository is a working fork of VideoMaMa, "Mask-Guided Video Matting via Generative Prior." It contains the upstream training and inference code plus a custom production UI/harness that combines SAM 3 mask tracking with VideoMaMa matting over image or EXR sequences.

Core pieces:

- `pipeline_svd_mask.py`: VideoMaMa inference pipeline built around Stable Video Diffusion with mask conditioning.
- `inference_onestep_folder.py`: batch folder inference entry point for pre-existing image and mask sequence folders.
- `demo/production_frame_app.py`: custom Gradio production UI for loading a sequence, adding SAM 3 keyframe prompts, generating masks, and running VideoMaMa over the shot.
- `demo/sam3_wrapper_hf.py`: production SAM 3 wrapper used by `production_frame_app.py`; supports multi-keyframe prompting and propagation.
- `demo/videomama_wrapper.py`: VideoMaMa wrapper used by the production app.
- `scripts/bootstrap_tmp_venv.sh`: builds local runtime venvs in `/tmp` and writes `.videomama-env`.
- `scripts/run_production_frame_ui.sh`: launches the production Gradio UI.

## Runtime Layout

Production machines keep code, weights, and venvs separate:

- Code/config: this repo, usually `/mnt/production/user/<username>/DEV/AI/VideoMama/`.
- Shared checkpoints: `/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints/`.
- Inference venv: `/tmp/videomama-venv/` using Python 3.9.
- SAM 3 UI venv: `/tmp/videomama-sam3-ui-venv/` using Python 3.12.

SAM 3 requires a newer Python/PyTorch/CUDA stack than the base VideoMaMa inference environment, so the UI and base inference environments are intentionally split.

Important environment variables are written to `.videomama-env`:

- `VIDEOMAMA_ROOT`
- `VIDEOMAMA_CHECKPOINTS`
- `VIDEOMAMA_VENV`
- `VIDEOMAMA_UI_VENV`
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
bash scripts/bootstrap_tmp_venv.sh sam3-ui
```

For `sam3-ui`, the bootstrap script auto-detects `python3.12` from `PATH` or a
pyenv-installed 3.12 under `~/.pyenv`. Override with `PYTHON_UI_BIN=/path/to/python`
when needed. The launch script activates the built UI venv; it does not create
the venv itself.

SAM 3 checkpoints are gated on Hugging Face. The UI venv must be authenticated
with `/tmp/videomama-sam3-ui-venv/bin/hf auth login`, and the token must have
accepted access to `facebook/sam3` or `facebook/sam3.1`. Override the default
with `SAM3_MODEL_VERSION=sam3.1` when launching.

The first SAM 3 Git commit resolved by the bootstrap is recorded in
`tmp/runtime-locks/sam3-requirement.txt` and reused for later `/tmp` venv
rebuilds. Set `REFRESH_SAM3_LOCK=1` (and optionally `SAM3_GIT_REF=<ref>`) when
intentionally updating SAM 3.

Launch the production SAM 3 + VideoMaMa UI:

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

The launch script pins `GRADIO_TEMP_DIR`/`GRADIO_EXAMPLES_CACHE` to absolute local
paths and runs from a stable cwd. This network mount can swap the working
directory out mid-session; Gradio resolves some cache paths relative to the cwd
and calls `getcwd()` per request, which otherwise 500s the file routes with
`FileNotFoundError`. `production_frame_app.py` sets the same defaults before
importing gradio, so running it directly is also safe.

## VRAM And Long Sequences

SAM 3 and VideoMaMa are co-resident on one GPU, so long shots can OOM in the
VideoMaMa matting pass (the VAE encode of a chunk is the usual peak). Mitigations
already wired in:

- `scripts/run_production_frame_ui.sh` exports `PYTORCH_ALLOC_CONF=expandable_segments:True`
  (and the legacy `PYTORCH_CUDA_ALLOC_CONF`) to reduce allocator fragmentation.
- `demo/production_frame_app.py` calls `torch.cuda.empty_cache()` after SAM 3 mask
  generation and after each VideoMaMa chunk.
- `pipeline_svd_mask.py` encodes the VAE in sub-batches; tune with
  `VIDEOMAMA_VAE_ENCODE_CHUNK` (default 4, lower if the encode still OOMs).

If you still OOM, lower the UI "VideoMaMa Chunk Size" (default 16) and/or
`VIDEOMAMA_VAE_ENCODE_CHUNK` before touching model code.

## Production UI Flow

`demo/production_frame_app.py` does the following:

1. Loads an image or EXR sequence from a directory.
2. Converts frames to RGB and caches resized 1024x576 JPEGs under `tmp/production_sequence_app/<timestamp>_<sequence>/sam3_frames`.
3. Lets the user add positive/negative SAM 3 point prompts on multiple keyframes.
4. Saves keyframe prompts to `keyframe_prompts.json`.
5. Uses `sam3_wrapper_hf.SAM3VideoTracker.track_video_from_dir()` to propagate masks across the cached sequence.
6. Saves masks under `sam3_masks`.
7. Runs VideoMaMa in chunks with configurable overlap.
8. Saves outputs under `videomama_frames` and grayscale alpha previews under `alpha_frames`.

Each run also writes `run_manifest.json` with source fingerprints, SAM prompt
provenance, VideoMaMa settings, per-frame completion records, and failure state.
VideoMaMa preview PNGs remain 8-bit for the UI; `alpha_frames` supports 16-bit
PNG, half-float EXR, or both. EXR model-input conversion supports gamma and
exposure controls.

The production app supports `.exr`, `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` inputs. EXR RGB conversion uses `OpenEXR`/`Imath`, `exr_gamma`, and exposure handling in code.

### Session Persistence And Resume

- The input panel (sequence directory, EXR gamma, point mode, chunk size, overlap,
  resume toggle) is saved to `tmp/production_sequence_app/ui_settings.json` on every
  change and reloaded as the defaults on the next launch.
- On "Load Sequence", if "Resume from tmp" is enabled the app looks for the most
  recent prior run of the same sequence under `tmp/production_sequence_app/` (matched
  via each run's `session.json`, or directory-name fallback for older runs) and
  restores its keyframe points (`keyframe_prompts.json`), generated masks
  (`sam3_masks/`), and VideoMaMa outputs (`videomama_frames/`). Cached 1024x576
  working frames are reused when the frame count and gamma match, so resume is fast.
- The app's working area is anchored to `<repo>/tmp/production_sequence_app/`
  (`APP_TMP_ROOT`) regardless of the launch directory.

### Processing A Frame Subrange

- "Process Start Frame" / "Process End Frame" inputs limit the "Run VideoMaMa
  (Selected Range)" pass to `[start, end]` inclusive. End `-1` (the default) means
  the last frame, so the whole sequence is selected unless narrowed.
- SAM 3 mask generation stays whole-sequence (one consistent propagation; masks
  persist), so generate masks once and then matte any range. Range runs reuse those
  masks and write per-frame outputs into the same `videomama_frames/` / `alpha_frames/`.
- Because outputs persist per frame, you can matte segment by segment (e.g. 0–15,
  then 12–27). Set the next range's start a few frames before the previous end; the
  chunk overlap re-processes those shared frames so the segment boundary blends.

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
- `scripts/requirements-sam3-ui.txt` is the Python 3.12 production UI dependency set and installs SAM 3 from GitHub.
- Torch is installed separately by `scripts/bootstrap_tmp_venv.sh`: CUDA 12.4
  wheels for the Python 3.9 inference venv and CUDA 12.8 wheels for the SAM 3 UI venv.
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
- Prefer editing only the custom production harness files when fixing SAM 3 + VideoMaMa UI behavior:
  - `demo/production_frame_app.py`
  - `demo/sam3_wrapper_hf.py`
  - `demo/videomama_wrapper.py`
  - `scripts/bootstrap_tmp_venv.sh`
  - `scripts/run_production_frame_ui.sh`
  - `scripts/requirements-*.txt`
- Be careful with `demo/sam2_wrapper.py` and `demo/sam2_wrapper_hf.py`: these are older SAM2 wrappers. The production app imports `sam3_wrapper_hf.py`.
- Generated app outputs belong under `tmp/production_sequence_app/` and should not be treated as source.
- Checkpoint directories can be very large; do not vendor weights into the repo.
- 1024x576 remains the production default. Higher 16:9 working resolutions are
  experimental and must remain divisible by 8; audit VRAM and matte quality when changing them.

## Useful Checks

Basic import/syntax checks without starting a GPU workload:

```bash
python -m py_compile demo/production_frame_app.py demo/sam3_wrapper_hf.py demo/videomama_wrapper.py
```

Check environment wiring:

```bash
source .videomama-env
printf '%s\n' "$VIDEOMAMA_CHECKPOINTS" "$VIDEOMAMA_BASE_MODEL_PATH" "$VIDEOMAMA_UNET_CHECKPOINT_PATH"
```

Starting the UI or running inference can require GPU access and large checkpoints. Do not assume those are available in every coding session.

### UI Regression Verification

- When fixing a production UI button, reproduce the failure against the actual
  loaded session/run path whenever the user reports one. Synthetic smoke tests are
  useful only after the real path has been exercised or the real failure has been
  encoded as a regression test.
- For "Clear Sequence Cache + Reload", verify the current `state['run_root']`
  directory is actually removed or moved aside on disk, and confirm the button
  reloads without resuming the stale cache. Network-mounted tmp directories may
  raise `OSError: [Errno 39] Directory not empty` during deletion; tests should
  cover that exact failure mode.

## Deployment Notes

- 2026-06-15: For shared studio use, prefer a private service rather than Gradio share: run the production UI on a G6 GPU host behind VPN/private ALB or reverse proxy with studio SSO/basic auth, mount shared storage/checkpoints, and serialize GPU jobs until a real queue/worker layer exists.
- For on-demand scaling, package the harness into a GPU container or AMI and run workers from a queue over shared S3/FSx/EFS storage; AWS Deadline Cloud may fit if the studio already manages VFX jobs there.

## Recent Production UI Changes

- 2026-06-15: `demo/production_frame_app.py` has a "Clear Sequence Cache + Reload" button that deletes the current per-sequence tmp run under `tmp/production_sequence_app/` and reloads without resuming.
- 2026-06-15: SAM 3 masks are still tracked on the 1024x576 UI/SAM working cache, but saved to `sam3_masks/` at each source frame's original resolution; previews downsample masks back to working size for overlay.
- 2026-06-15: `demo/videomama_wrapper.py` resizes each VideoMaMa output back to its matching source frame size, and the production app guards output saves to source resolution.
- 2026-06-15: The production UI now letterboxes source frames into the fixed 1024x576 SAM/UI canvas instead of stretching; SAM masks are unletterboxed back to source resolution when saved. Prompt clicks in letterbox padding are rejected.
- 2026-06-15: The production UI includes a large prompting accordion with its own scrub slider and fullscreen-enabled image for more accurate artist point picking.
