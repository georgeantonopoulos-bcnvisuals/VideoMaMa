---
title: VideoMaMa - Video Matting with Mask Guidance
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎬 VideoMaMa: Video Matting with Mask Guidance

The repository contains two interfaces:

- `app.py` is the original uploaded-video SAM2 demo.
- `production_frame_app.py` is the studio sequence harness. It combines SAM 3 / 3.1
  point/text prompting and propagation with a selectable matting backend over image
  or EXR shots: SAM2Matting SAM2.1 Base+ by default, VideoMaMa as an alternative
  and rescue path. See `../docs/sam2matting.md`.

For production rotoscoping, use `production_frame_app.py` through
`scripts/run_production_frame_ui.sh`.

## 🌟 Features

- **Single-Click Object Selection**: Simply click on the object you want to extract in the first frame
- **Automatic Tracking**: SAM2 automatically tracks your selected object through all frames
- **High-Quality Matting**: VideoMaMa generates smooth, temporally-consistent alpha mattes
- **Flexible Input**: Upload your own video or try our provided samples
- **Customizable**: Adjust augmentation settings for different scenarios

## 🚀 Production Sequence Workflow

0. Runtimes live under `$VIDEOMAMA_VENV_ROOT` (default `/mnt/temporal/VideoMama`),
   not `/tmp` — `/tmp` is the root filesystem on these hosts and too small.
1. Bootstrap with `bash scripts/bootstrap_tmp_venv.sh everything` and authenticate
   the UI venv for the gated `facebook/sam3` or `facebook/sam3.1` checkpoint. The
   `everything` target also builds the isolated SAM2Matting runtime; run
   `bash scripts/bootstrap_sam2matting.sh` on its own to (re)build just that.
1b. Pick the **Tracking Model** (`sam3` / `sam3.1`) and the **Matting Backend**.
   Leave **Matte ROI** on `Auto (from SAM 3 masks)` unless you want to frame the
   subject yourself. Note that SAM2Matting always runs its network at 1024px
   square — the ROI, not the source resolution, is what buys edge detail.
2. Launch `bash scripts/run_production_frame_ui.sh`.
3. Optionally open **SAM Source ROI Crop**, load a full-resolution selector frame,
   and click two opposite corners. The fixed source ROI is cropped before it is
   resized to the SAM canvas, increasing effective detail on the subject.
4. Load the image/EXR sequence directory and add point keyframes or a text concept.
5. Generate SAM 3 masks for the full sequence.
6. Run VideoMaMa over the full shot or an inclusive frame range.
7. Review 8-bit previews under `videomama_frames` and use the production alpha
   from `alpha_frames` (16-bit PNG, half EXR, or both).

Runs are saved under `tmp/production_sequence_app`, with a manifest recording
source fingerprints, prompts, model settings, generated frames, and completion
status. Chunk overlaps are cross-faded to reduce visible range boundaries.
EXR previews may use the simple exposure/gamma transform or the server's active
`OCIO` configuration with an input colorspace, display, and view.
When an ROI is enabled, both SAM 3 and VideoMaMa process that source crop at the
selected working resolution; their masks/mattes are then mapped back into
full-resolution, full-frame outputs. This avoids shrinking the entire 4K plate
when only a smaller subject region needs detail.

The combined quality preset affects both stages: its threshold changes SAM 3,
while its processing resolution and plate-edge refinement change VideoMaMa.
`Maximum Detail` selects 2048x1152 processing and is the highest-VRAM option.
`Hair Detail` also uses 2048x1152, preserves SAM 3's raw irregular boundary,
disables both plate-smoothing passes, and expands only the VideoMaMa conditioning
guide by 8 source pixels so nearby wisps are not excluded before matting.

## 🚀 Legacy Uploaded-Video Demo

1. **Upload a video** or **select from samples**
2. **Click on the object** you want to extract in the first frame (displayed in the interface)
3. Optionally adjust **augmentation settings** in the advanced options
4. Click **"Generate Matting"** and wait for processing
5. View your results: output video, comparison images, and mask track


## 🔧 Installation (Local Setup)

If you want to run this demo locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Add sample videos to samples/ directory (optional)

# Run the demo
python app.py
```

## 🎯 Tips for Best Results

- **Click Precisely**: Click on the center of the object you want to extract
- **Clear Objects**: Works best with distinct foreground objects
- **Video Length**: For faster processing, use shorter videos (< 5 seconds)
- **Augmentations**: 
  - Use "polygon" for cleaner geometric masks
  - Enable temporal augmentation for challenging videos
  - Try "bounding box" for very simple selections

## 📚 Technical Details

### Model Architecture
- **Base Model**: Stable Video Diffusion (SVD-XT)
- **Conditioning**: RGB frames + VAE-encoded masks
- **UNet**: Fine-tuned with additional mask conditioning channels
- **Processing**: Chunked inference (16 frames per chunk)

### SAM Integration
- The production sequence harness uses SAM 3 with multi-keyframe point prompts
  or text concepts.
- The legacy uploaded-video demo continues to use SAM2.

## 🤝 Contributing

If you encounter issues or have suggestions:
1. Check that all model checkpoints are correctly placed
2. Ensure your GPU has sufficient VRAM
3. Try reducing video length or resolution for testing


## 🙏 Acknowledgments

- **SAM2**: Meta AI's Segment Anything 2
- **Stable Video Diffusion**: Stability AI's video generation model
- **Gradio**: For the amazing UI framework

## 📧 Contact

For questions or issues, please open an issue on our GitHub repository.

---

**Note**: This demo is for research purposes. Processing times may vary based on video length and available compute resources.
