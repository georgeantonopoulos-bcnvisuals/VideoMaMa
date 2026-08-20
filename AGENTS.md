# AGENTS.md

## Project Overview

This repository is a working fork of VideoMaMa, "Mask-Guided Video Matting via Generative Prior." It contains the upstream training and inference code plus a custom production UI/harness that combines SAM 3 mask tracking with a selectable matting backend over image or EXR sequences.

SAM 3 / 3.1 does tracking. The matte itself comes from a **matting backend**:
SAM2Matting SAM2.1 Base+ by default, with VideoMaMa retained as an alternative
and rescue path. See `docs/sam2matting.md` for the full design, validation
results and the A/B comparison procedure.

Core pieces:

- `pipeline_svd_mask.py`: VideoMaMa inference pipeline built around Stable Video Diffusion with mask conditioning.
- `inference_onestep_folder.py`: batch folder inference entry point for pre-existing image and mask sequence folders.
- `demo/production_frame_app.py`: custom Gradio production UI for loading a sequence, adding SAM 3 keyframe prompts, generating masks, and running VideoMaMa over the shot.
- `demo/sam3_wrapper_hf.py`: production SAM 3 wrapper used by `production_frame_app.py`; supports multi-keyframe prompting and propagation.
- `demo/videomama_wrapper.py`: VideoMaMa wrapper used by the production app.
- `demo/matting_backends.py`: matting backend registry, parameters, conditioning strategy, and cache identity. Stdlib only.
- `demo/subject_roi.py`: automatic stable source-space ROI derived from generated SAM 3 masks. Pure numpy.
- `demo/matte_qc.py`: alpha-vs-SAM-mask disagreement metrics used to flag suspicious ranges. Pure numpy.
- `demo/alpha_transforms.py`: ROI/letterbox <-> source alpha mapping, overlap cross-fade, QC downsampling. Pure numpy.
- `demo/sam2matting_client.py`: UI-side driver for the SAM2Matting worker (staging, subprocess, NDJSON streaming).
- `demo/sam2matting_worker.py`: runs **only** inside the isolated SAM2Matting venv. Never import it from the UI.
- `scripts/bootstrap_tmp_venv.sh`: builds local runtime venvs in `/tmp` and writes `.videomama-env`.
- `scripts/bootstrap_sam2matting.sh`: builds the isolated SAM2Matting venv, pinned checkout, and shared checkpoints.
- `scripts/run_production_frame_ui.sh`: launches the production Gradio UI.

## Runtime Layout

Production machines keep code, weights, and venvs separate:

- Code/config: this repo, usually `/mnt/production/user/<username>/DEV/AI/VideoMama/`.
- Shared checkpoints: `/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints/`.
- Runtimes: `$VIDEOMAMA_VENV_ROOT` (default `/mnt/temporal/VideoMama`, a 419 GB local NVMe).
  - Inference venv: `videomama-venv/` using Python 3.9.
  - SAM 3 UI venv: `videomama-sam3-ui-venv/` using Python 3.12 and torch 2.10.
  - SAM2Matting venv: `videomama-sam2matting-venv/` using Python 3.12 and torch 2.8.
  - SAM2Matting source: `videomama-sam2matting-src/`, detached at a pinned commit.

Do not put these under `/tmp`: on these hosts `/tmp` is the root filesystem and
runs out of space long before three CUDA venvs fit. Relocate the whole set with
`VIDEOMAMA_VENV_ROOT`, or pin one with `VIDEOMAMA_VENV` / `VIDEOMAMA_UI_VENV` /
`VIDEOMAMA_SAM2MATTING_VENV`. The venvs survive reboots, so rebuild only when
dependencies change.

SAM 3 requires a newer Python/PyTorch/CUDA stack than the base VideoMaMa inference environment, so the UI and base inference environments are intentionally split.

**Never install SAM2Matting into the SAM 3 UI venv.** SAM2Matting vendors modified
top-level `sam2` and `sam3` packages and expects torch 2.8; sharing an environment
would shadow the official `facebookresearch/sam3` install the UI depends on. It gets
its own venv, its own source checkout, and its own process, driven over a JSON job
file and NDJSON on stdout by `demo/sam2matting_client.py`.

Important environment variables are written to `.videomama-env`:

- `VIDEOMAMA_ROOT`
- `VIDEOMAMA_CHECKPOINTS`
- `VIDEOMAMA_VENV_ROOT`
- `VIDEOMAMA_VENV`
- `VIDEOMAMA_UI_VENV`
- `VIDEOMAMA_BASE_MODEL_PATH`
- `VIDEOMAMA_UNET_CHECKPOINT_PATH`
- `VIDEOMAMA_SAM2MATTING_VENV`
- `VIDEOMAMA_SAM2MATTING_HOME`

The SAM2Matting runtime is also recorded in `tmp/runtime-locks/sam2matting-runtime.json`
(upstream commit, checkpoint revision and sha256 digests, torch version), which the
app reads as a fallback and folds into every matte's cache identity.

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
bash scripts/bootstrap_tmp_venv.sh sam2matting   # delegates to bootstrap_sam2matting.sh
bash scripts/bootstrap_tmp_venv.sh everything    # all three
```

The SAM2Matting runtime can also be built directly, with sub-targets:

```bash
bash scripts/bootstrap_sam2matting.sh              # source + venv + checkpoints + lock
bash scripts/bootstrap_sam2matting.sh source       # pinned checkout only
bash scripts/bootstrap_sam2matting.sh env          # venv only
bash scripts/bootstrap_sam2matting.sh checkpoints  # weights only
```

SAM2Matting checkpoints are public and land in
`$VIDEOMAMA_CHECKPOINTS/SAM2Matting/`. They are never vendored into Git. Moving
the upstream pin requires `SAM2MATTING_GIT_REF=<sha> REFRESH_SAM2MATTING_LOCK=1`
and a matching update to `SAM2MATTING_PINNED_COMMIT` in `demo/matting_backends.py`;
the script refuses to silently record a different commit.

For `sam3-ui`, the bootstrap script auto-detects `python3.12` from `PATH` or a
pyenv-installed 3.12 under `~/.pyenv`. Override with `PYTHON_UI_BIN=/path/to/python`
when needed. The launch script activates the built UI venv; it does not create
the venv itself.

SAM 3 checkpoints are gated on Hugging Face. The UI venv must be authenticated
with `$VIDEOMAMA_UI_VENV/bin/hf auth login`, and the token must have
accepted access to `facebook/sam3` or `facebook/sam3.1`. Override the default
with `SAM3_MODEL_VERSION=sam3.1` when launching.

The first SAM 3 Git commit resolved by the bootstrap is recorded in
`tmp/runtime-locks/sam3-requirement.txt` and reused for later venv
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

SAM2Matting is different: it runs in its own process, so the UI unloads SAM 3 and
VideoMaMa before launching the worker and the worker's memory is reclaimed when it
exits. Its own footprint is kept flat by:

- CPU offload of the decoded video and the tracking state (UI: "CPU Offload").
- Pruning the per-frame alpha upstream stores in each frame's output dict and
  never moves off the GPU. Measured at 1024x1024: 1187 MiB peak over 120 frames
  with pruning versus 1656 MiB without, and 1259 MiB over 400 frames.

If SAM2Matting OOMs, in order: keep CPU offload on, lower "Frame Cap", lower
"Window Size", tighten the matte ROI, then fall back to the Tiny variant.

The worker also shims a real upstream bug: with `offload_video_to_cpu=True`,
`_run_single_frame_inference` feeds `inference_state["images"][idx]` to the alpha
heads without moving it to the compute device. `_DeviceFrameView` in
`demo/sam2matting_worker.py` serves one frame at a time on the GPU. Do not
"fix" this by disabling the offload.

## Production UI Flow

`demo/production_frame_app.py` does the following:

1. Loads an image or EXR sequence from a directory.
2. Converts frames to RGB and caches resized 1024x576 JPEGs under `tmp/production_sequence_app/<timestamp>_<sequence>/sam3_frames`.
3. Lets the user add positive/negative SAM 3 point prompts on multiple keyframes.
4. Saves keyframe prompts to `keyframe_prompts.json`.
5. Uses `sam3_wrapper_hf.SAM3VideoTracker.track_video_from_dir()` to propagate masks across the cached sequence.
6. Saves masks under `sam3_masks` at source resolution.
7. Resolves the **matte ROI** (auto from those masks, manual SAM crop override, or full frame).
8. Runs the selected **matting backend** over the requested range.
9. Saves outputs under that backend's directories and grayscale alpha previews alongside them.

### Matting Backends

- `SAM2Matting SAM2.1 Base+` (default) -> `matte_frames_s2m_base_plus/`, `alpha_frames_s2m_base_plus/`
- `VideoMaMa` -> `videomama_frames/`, `alpha_frames/` (unchanged, still the rescue/comparison path)
- `SAM2Matting SAM2.1 Tiny` and `SAM2Matting SAM3 tracker` -> experimental, own directories

Per-backend directories exist so an A/B never overwrites its own baseline. Do not
apply the GrabCut / guided-filter refinements to SAM2Matting output: predicting
soft boundaries is what it is for, and those operations remove hair.

SAM2Matting's neural inference is 1024x1024 square with a 512x512 alpha head
(1008/504 for the vendored-SAM3 variant). Feeding it a 4K frame does not make
inference 4K. The lever is the matte ROI, not the input resolution; never claim
otherwise in UI text or docs.

### Matte ROI

Resolution order: `Full frame` -> an enabled manual SAM ROI crop (the artist's
override always wins) -> `Auto (from SAM 3 masks)`. Auto takes the union of the
generated masks over the processed range, pads it, limits its aspect, and holds
it **fixed for the whole range** so the matte cannot breathe. If it saves less
than 1.35x in area it is dropped in favour of the full frame.

### Conditioning

Upstream's demo conditions on frame 0 only. That was measured to lose the object
by frame 2 on a real 4K plate, so the default is artist keyframes plus a periodic
SAM 3 re-anchor (every 4 frames) plus the range endpoints. `Dense` conditions on
every frame. Anchoring is nearly free; if QC flags frames, lower the interval
before changing anything else.

Each run also writes `run_manifest.json` with source fingerprints, SAM prompt
provenance, the matte section's backend identity and settings, per-frame
completion records, and failure state.

The `matte` section records backend id, upstream commit, checkpoint name,
checkpoint revision and sha256, torch version, ROI geometry and every matting
parameter, all of which feed `settings_hash`. Frame records are only reused when
their `settings_hash` matches, so a Base+ matte can never be served as a
VideoMaMa one. The VideoMaMa backend also mirrors its result into the legacy
`videomama` section so older runs stay resumable. `matte_qc_<backend>.json` holds
per-frame QC metrics and flagged ranges.
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
- Prefer editing only the custom production harness files when fixing UI behavior:
  - `demo/production_frame_app.py`
  - `demo/sam3_wrapper_hf.py`
  - `demo/videomama_wrapper.py`
  - `demo/matting_backends.py`, `demo/subject_roi.py`, `demo/matte_qc.py`, `demo/alpha_transforms.py`
  - `demo/sam2matting_client.py`, `demo/sam2matting_worker.py`
  - `scripts/bootstrap_tmp_venv.sh`, `scripts/bootstrap_sam2matting.sh`
  - `scripts/run_production_frame_ui.sh`
  - `scripts/requirements-*.txt`
- `production_frame_app.py` is already large. New matting logic belongs in the
  small modules above, not in the app.
- Never import `demo/sam2matting_worker.py` (or anything from the SAM2Matting
  checkout) into the UI process. The isolation is the design.
- Be careful with `demo/sam2_wrapper.py` and `demo/sam2_wrapper_hf.py`: these are older SAM2 wrappers. The production app imports `sam3_wrapper_hf.py`.
- Generated app outputs belong under `tmp/production_sequence_app/` and should not be treated as source.
- Checkpoint directories can be very large; do not vendor weights into the repo.
- 1024x576 remains the production default. Higher 16:9 working resolutions are
  experimental and must remain divisible by 8; audit VRAM and matte quality when changing them.

## Useful Checks

Basic import/syntax checks without starting a GPU workload:

```bash
python -m py_compile demo/*.py
```

Run the test suite (no GPU required; the SAM2Matting worker is faked):

```bash
"$VIDEOMAMA_UI_VENV/bin/python" -m unittest tests.test_production_frame_app tests.test_sam2matting_integration
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

- 2026-08-20: Runtimes moved off `/tmp` to `$VIDEOMAMA_VENV_ROOT`
  (default `/mnt/temporal/VideoMama`). `/tmp` is the root filesystem here and
  cannot hold three CUDA venvs; the move is what made the SAM 3 UI venv
  buildable on this host at all.
- 2026-08-20: The matte preview tab is now **Matte**, not "VideoMaMa", and its
  label and the Frame Info line name the backend that produced what is on
  screen. A SAM2Matting result previously rendered under a VideoMaMa heading and
  looked like it had never been generated.
- 2026-08-20: Added a matting-backend layer. SAM 3 / 3.1 remains the tracker; the
  matte comes from SAM2Matting SAM2.1 Base+ by default, with VideoMaMa retained
  unchanged as an alternative and rescue path. Adds an automatic stable subject
  ROI from the SAM 3 masks, backend-aware manifests and cache invalidation, and
  lightweight QC. SAM2Matting runs in an isolated venv and process, pinned to
  upstream `73dd721d`. Full design, measurements and the A/B procedure are in
  `docs/sam2matting.md`.

- 2026-06-15: `demo/production_frame_app.py` has a "Clear Sequence Cache + Reload" button that deletes the current per-sequence tmp run under `tmp/production_sequence_app/` and reloads without resuming.
- 2026-06-15: SAM 3 masks are still tracked on the 1024x576 UI/SAM working cache, but saved to `sam3_masks/` at each source frame's original resolution; previews downsample masks back to working size for overlay.
- 2026-06-15: `demo/videomama_wrapper.py` resizes each VideoMaMa output back to its matching source frame size, and the production app guards output saves to source resolution.
- 2026-06-15: The production UI now letterboxes source frames into the fixed 1024x576 SAM/UI canvas instead of stretching; SAM masks are unletterboxed back to source resolution when saved. Prompt clicks in letterbox padding are rejected.
- 2026-06-15: The production UI includes a large prompting accordion with its own scrub slider and fullscreen-enabled image for more accurate artist point picking.
