# SAM3 (+3.1) tracking with SAM2Matting mattes

This document covers the matting-backend layer added on top of the existing
SAM 3 production roto pipeline: what it is, how to bootstrap it, what it was
validated against, and how to compare SAM2Matting with VideoMaMa on the same
frames.

## 1. Architecture and why

### The shape of the problem

SAM 3 is a very good tracker and a mediocre matte source: it produces a hard
binary mask. VideoMaMa turns that mask into a soft alpha by running a
Stable-Video-Diffusion UNet, which is slow, seeded, and chunked. SAM2Matting
attacks the same problem directly — a VOS tracker plus a dedicated ROI
detection and progressive alpha head trained for matting.

Two constraints shaped everything else:

1. **SAM2Matting cannot share our SAM 3 environment.** Upstream ships modified
   top-level `sam2` and `sam3` packages and expects torch 2.8, while the
   production UI runs the official `facebookresearch/sam3` on torch 2.10.
   Installing them together would let SAM2Matting's `sam3` shadow ours.
2. **SAM2Matting's neural inference is 1024×1024, always.** `load_video_frames`
   resizes every frame to a *square* 1024, and the alpha decoder emits 512×512
   before being interpolated up. Handing it a 3840×2160 plate does not make
   inference 4K; it just throws subject detail away before the network runs.

### What was built

| Piece | Role |
|---|---|
| `demo/matting_backends.py` | Backend registry, parameters, conditioning strategy, and the identity/cache-key rules. Stdlib only. |
| `demo/subject_roi.py` | Derives one fixed, padded source-space ROI from the generated SAM 3 masks. Pure numpy. |
| `demo/matte_qc.py` | Compares alpha support against SAM 3 mask support and flags suspicious ranges. Pure numpy. |
| `demo/alpha_transforms.py` | ROI/letterbox to source alpha mapping, overlap cross-fade, QC downsampling. Pure numpy. |
| `demo/sam2matting_client.py` | Runs in the UI process: stages inputs, launches the worker, streams NDJSON, yields float32 alpha. |
| `demo/sam2matting_worker.py` | Runs **only** in the isolated SAM2Matting venv. Never imported by the UI. |
| `scripts/bootstrap_sam2matting.sh` | Builds the isolated venv, the pinned source checkout, and the shared checkpoints. |

`demo/production_frame_app.py` keeps orchestration (SAM 3 masks, ranges,
manifests, resume, output formats) and dispatches the matting pass to one of two
functions, `_run_videomama_pass` or `_run_sam2matting_pass`, that share a
`_MatteSink` for overlap cross-fading and writing.

### Process isolation

The worker is a subprocess launched with `PYTHONPATH` cleared, `cwd` set to the
pinned checkout, and communication over a JSON job file plus NDJSON on stdout.
Alpha crosses the boundary as float32 `.npy` files that are deleted as they are
consumed. Before the worker starts, the UI unloads SAM 3 and VideoMaMa so the
worker has the whole GPU.

### Conditioning strategy

Upstream's demo conditions on a mask at frame 0 and propagates. That is not good
enough here, and we measured why (§4): on a 4K plate with thin branches,
first-frame-only conditioning lost the object completely by frame 2.

Our system already has a full set of SAM 3 masks and artist-authored keyframes,
so the default re-anchors on them:

- **Artist keyframes always condition.** They are the corrections that were paid
  for, and `propagate_in_video` re-emits a conditioned frame's alpha verbatim,
  so an artist correction survives exactly.
- **Plus periodic SAM 3 anchors** (default every 4 frames) and the range
  endpoints. Anchoring is nearly free, and it bounds identity drift.
- **Dense** (every frame) is available for hard shots; it removes drift entirely
  at the cost of the tracker's temporal smoothing.

Every window also conditions on its own first frame, which keeps the worker's
propagation forward-only and its output strictly monotonic in frame index.

### ROI

The **matte ROI** is separate from the existing load-time SAM ROI crop, because
it is derived from masks that do not exist until SAM 3 has run. Resolution order:

1. `Full frame` — never crop.
2. An enabled manual SAM ROI crop — the artist's explicit override wins.
3. `Auto (from SAM 3 masks)` — union of the masks over the processed range,
   padded (12% of the subject box, floor 24 px), aspect-limited, aligned, and
   **held fixed for the whole range** so the sampling grid never moves. A moving
   ROI makes hair edges crawl.

If the resulting ROI saves less than 1.35× in area, it is dropped and the frame
is processed whole.

### Precision and post-processing

Alpha is float32 from the worker, through the ROI inverse mapping, into the
existing 16-bit PNG / half-EXR writers. The only 8-bit step is the UI preview.
The GrabCut and guided-filter refinements are **not** applied to SAM2Matting
output: predicting soft boundaries is exactly what it is for, and those
operations remove hair.

### Memory

Two things keep a long 4K shot inside a 24 GB L4:

- CPU offload of both the decoded video and the tracking state.
- **Pruning the stored per-frame alpha.** Upstream keeps `alpha` at full video
  resolution inside each frame's output dict and, unlike `maskmem`, never moves
  it to the storage device. The worker drops it once written. Measured on 120
  frames at 1024²: 1187 MiB with pruning, 1656 MiB without (~4 MiB/frame). At
  400 frames, pruning keeps the peak at 1259 MiB — flat.

The worker also carries a shim for a genuine upstream bug: with
`offload_video_to_cpu=True`, `_run_single_frame_inference` reads
`inference_state["images"][idx]` and feeds it to the alpha heads without moving
it to the compute device, so it crashes with a device mismatch. `_DeviceFrameView`
serves one frame at a time on the GPU while the rest stays in host RAM.

Compilation and TensorRT are deliberately **not** wired up. Eager BF16 is
correct and fast enough; `--compiled` remains available upstream for later work.

## 2. Bootstrap and run

```bash
# Core environments (unchanged)
bash scripts/bootstrap_tmp_venv.sh

# Isolated SAM2Matting runtime: venv + pinned checkout + shared checkpoints
bash scripts/bootstrap_sam2matting.sh
# or, as part of the normal bootstrap:
bash scripts/bootstrap_tmp_venv.sh everything

source .videomama-env
bash scripts/run_production_frame_ui.sh
```

Sub-targets: `source`, `env`, `checkpoints`, `lock`.

Layout:

All runtimes live under `$VIDEOMAMA_VENV_ROOT`, default `/mnt/temporal/VideoMama`
(a 419 GB local NVMe). They are deliberately *not* under `/tmp`, which on these
hosts is the root filesystem and too small for three CUDA venvs.

| What | Where |
|---|---|
| SAM2Matting venv | `$VIDEOMAMA_VENV_ROOT/videomama-sam2matting-venv` (torch 2.8.0+cu128, Python 3.12) |
| SAM2Matting source | `$VIDEOMAMA_VENV_ROOT/videomama-sam2matting-src`, detached at the pinned commit |
| Checkpoints | `$VIDEOMAMA_CHECKPOINTS/SAM2Matting/` (shared storage, never in Git) |
| Runtime record | `tmp/runtime-locks/sam2matting-runtime.json` |
| Pip lock | `tmp/runtime-locks/requirements-sam2matting.lock.txt` |

Moving the upstream pin is deliberate and refuses to happen by accident:

```bash
SAM2MATTING_GIT_REF=<sha> REFRESH_SAM2MATTING_LOCK=1 bash scripts/bootstrap_sam2matting.sh source
# then update SAM2MATTING_PINNED_COMMIT in demo/matting_backends.py
```

In the UI: pick **Tracking Model** (`sam3` / `sam3.1`), pick **Matting Backend**,
set the **Matte ROI** mode, prompt keyframes, *Generate SAM 3 Masks*, then
*Generate Matte (Selected Range)*. Ranges and resume work exactly as before.

## 3. Exact model and checkpoint variants

| Backend | Checkpoint | sha256 (first 16) | Model res | Alpha head |
|---|---|---|---|---|
| SAM2Matting SAM2.1 Base+ **(default)** | `SAM2Matting-SAM2.1Base+.pt` (366 MB) | `1f0eb2eda3e8bc91` | 1024² | 512² |
| SAM2Matting SAM2.1 Tiny (experimental) | `SAM2Matting-SAM2.1Tiny.pt` (206 MB) | `5b9321e3b51bc20f` | 1024² | 512² |
| SAM2Matting SAM3 tracker (experimental) | `SAM2Matting-SAM3.pt` (3.3 GB) | `7102d695be6070b3` | 1008² | 504² |
| VideoMaMa | `$VIDEOMAMA_BASE_MODEL_PATH` + `$VIDEOMAMA_UNET_CHECKPOINT_PATH` | — | UI processing resolution | — |

- Upstream code: `FudanCVL/SAM2Matting` @ `73dd721d77b56749248aefe5e8824d7f61b9d13c`
- Checkpoint repo: `FudanCVL/SAM2Matting` @ `4315db9c60d27fde396b09765748a0ca6c97bed5`
- Worker torch: `2.8.0+cu128`, Python 3.12.13

The vendored-SAM3 variant is self-contained: every weight comes from
`SAM2Matting-SAM3.pt`, so it needs no gated Hugging Face download and is
unrelated to our official SAM 3 install.

Each matte's manifest records the backend id, upstream commit, checkpoint name,
checkpoint revision, checkpoint sha256, torch version, ROI geometry, and every
matting parameter. All of it feeds the `settings_hash`, so results from different
backends, revisions, ROIs or parameters can never be reused for each other.

## 4. What was measured, and what was not

### Measured on this machine (NVIDIA L4 24 GB, Rocky 9)

Real source: `uni_island/.../render_shot_080_plt_retime30fps/v001`, 25 frames of
3840×2160 EXR, with the SAM 3 masks from an earlier real production run.

- **Base+ end to end, 25 frames**: 43.7 s (1.75 s/frame) including 4K EXR decode,
  JPEG staging, model load, inference and writing both 16-bit PNG and half EXR at
  full 3840×2160. Auto ROI 1560×2016, 2.64× area gain.
- **Windowed** (window 10, overlap 4) over the same 25 frames: 65.6 s, all frames
  written, manifest complete.
- **Manual ROI override**: honoured exactly (2325×1641 at 752,269), reported as
  `applied: manual`.
- **Tiny variant**: 8 frames in 16.4 s, distinct checkpoint digest in the manifest.
- **Vendored-SAM3 variant**: 4 frames, completes, produces more soft pixels than
  Base+ on this shot. Still labelled experimental.
- **VRAM** (synthetic 1024², worker-reported peak): 1187 MiB at 120 frames,
  1259 MiB at 400 frames. Throughput ~8.5 fps at 1024².
- **Conditioning strategy** (12 frames of the real shot):

  | Strategy | Mean alpha coverage vs mask | QC-flagged | Max frame-to-frame Δalpha | Wall clock |
  |---|---|---|---|---|
  | Artist keyframes only | 0.17 (object lost from frame 2) | 10 of 12 | — | 22.2 s |
  | Periodic, every 8 | 0.83 | 1 | 0.0309 | 23.3 s |
  | Periodic, every 4 (**default**) | 0.82 | 1 | 0.0332 | 23.6 s |
  | Periodic, every 2 | 0.96 | 0 | 0.0245 | 24.0 s |
  | Dense (every frame) | 0.90 | 0 | 0.0208 | 24.8 s |

  Conditioning density costs almost nothing. On this shot, interval 2 was best on
  both accuracy and worst-case flicker; the default of 4 is a compromise chosen
  from a single shot's evidence. **If QC flags frames, lower the interval first.**

### Full pipeline, driven through the UI

After relocating the runtimes to `/mnt/temporal/VideoMama`, the whole chain was
exercised in a browser against `scripts/run_production_frame_ui.sh` on the same
4K shot, with the artist's real keyframe re-projected onto a 1024x576 canvas:

- **SAM 3 loaded and tracked for real**: 25 masks propagated from 1 keyframe,
  `frames_with_pixels=25/25`, 8.6 GB VRAM while resident.
- **SAM2Matting** matted frames 0-3 from that live mask set, after the app
  unloaded SAM 3 to free the GPU. QC clean.
- **VideoMaMa** matted the identical frames from the identical masks and ROI.

A/B on frames 0-3 (half-EXR alpha, same SAM 3 masks, same manual ROI):

| frame | SAM2Matting soft px | VideoMaMa soft px | SAM2Matting mass | VideoMaMa mass | mean abs diff |
|---|---|---|---|---|---|
| 1001 | 79,761 | 184,552 | 533,688 | 494,077 | 0.0106 |
| 1002 | 69,526 | 116,483 | 677,397 | 424,797 | 0.0340 |
| 1003 | 73,297 | 112,689 | 672,473 | 416,844 | 0.0343 |
| 1004 | 79,419 | 157,621 | 647,443 | 648,353 | 0.0059 |

VideoMaMa reports far more fractional pixels, but visual inspection shows that
softness spread across the *interior* of the subject rather than concentrated at
its boundary — generative blur, not recovered hair. It also loses roughly a third
of the alpha mass on frames 1002-1003 where SAM2Matting holds coverage. On this
shot SAM2Matting is the better matte; VideoMaMa remains useful as a second
opinion on ranges where the tracker struggles. One shot is not a general verdict.

### Not validated here

- **`sam3.1` weights** were not exercised; `facebook/sam3` was used throughout.
  The selector sets the version the existing wrapper already supports and
  switching it is unit-tested, but no `facebook/sam3.1` run was performed.
- **Long-shot behaviour beyond 400 frames** was extrapolated from the flat memory
  curve, not measured.
- **Compilation / TensorRT** was not attempted, by design.
- QC thresholds were tuned against one shot. Treat flagged ranges as "look here
  first", not as a verdict.

## 5. Comparing SAM2Matting against VideoMaMa on the same frames

Both backends read the same SAM 3 masks and write to their own directories inside
one run root, so a comparison never overwrites its own baseline.

1. Load the sequence and generate SAM 3 masks once. Every backend reuses them.
2. Choose a range worth arguing about (a hair or motion-blur heavy handful of
   frames), e.g. Start 40 / End 55.
3. Run **SAM2Matting SAM2.1 Base+** over that range. Outputs land in
   `matte_frames_s2m_base_plus/` and `alpha_frames_s2m_base_plus/`.
4. Switch **Matting Backend** to **VideoMaMa** and run the same range. Outputs land
   in `videomama_frames/` and `alpha_frames/`.
5. Compare `alpha_frames_s2m_base_plus/` against `alpha_frames/` in Nuke, in the
   UI's **Matte** tab (its label and the Frame Info line name the backend that
   produced the frame on screen, so the two runs cannot be confused), or with
   the snippet below. Read `run_manifest.json` → `matte` for the settings that
   produced the current result, and `matte_qc_*.json` for the per-frame metrics.

Keep `Matte ROI` and `Alpha Output Format` identical across the two runs so the
only variable is the matting model.

```bash
"$VIDEOMAMA_UI_VENV/bin/python" - <<'PY'
import numpy as np
from pathlib import Path
from PIL import Image

run = Path("tmp/production_sequence_app/<run>")
a = run / "alpha_frames_s2m_base_plus"   # SAM2Matting
b = run / "alpha_frames"                 # VideoMaMa

for path in sorted(a.glob("*.png")):
    other = b / path.name
    if not other.is_file():
        continue
    x = np.array(Image.open(path)).astype(np.float32) / 65535.0
    y = np.array(Image.open(other)).astype(np.float32) / 65535.0
    soft = lambda m: int(np.count_nonzero((m > 0.02) & (m < 0.98)))
    print(f"{path.stem}: mean|diff|={np.abs(x - y).mean():.5f} "
          f"soft_px s2m={soft(x)} vm={soft(y)} "
          f"mass s2m={x.sum():.0f} vm={y.sum():.0f}")
PY
```

`soft_px` is the useful column: it counts genuinely fractional alpha, which is
where hair and motion blur live and where the two models differ most.

To rescue a difficult range with VideoMaMa, narrow Start/End to that range and
run it with the VideoMaMa backend. Because per-frame records are keyed by
`settings_hash`, the rest of the shot is untouched and the manifest keeps an
honest record of which model produced which frames.
