### Directory Structure

The script expects a specific directory structure for your input data. You need a root folder for images and another for masks. Inside each, there should be subdirectories for each video, with matching names.

```
/path/to/your/dataset/
├── image/
│   ├── video_001/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── video_002/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│
└── mask/
    ├── video_001/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── video_002/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
```

* The names of the image files and mask files within the corresponding video folders must match (e.g., `0000.png` in images corresponds to `0000.png` in masks).

### Usage

Run the script from your terminal. Below is the general usage format and a detailed breakdown of all available arguments.

#### Basic Command

```bash
python inference_onestep_folder.py \
    --base_model_path "/path/to/your/stable-video-diffusion-img2vid-xt" \
    --unet_checkpoint_path "/path/to/your/unet_checkpoint" \
    --image_root_path "/path/to/image/folder" \
    --mask_root_path "/path/to/mask/folder" \
    --output_dir "/path/to/save/results" \
    [--optional_arguments]
```

#### Command-Line Arguments

##### **Paths**
* `--base_model_path`: Path to the base SVD model directory. (Default: `/path/to/pretrained_models/stable-video-diffusion-img2vid-xt`)
* `--unet_checkpoint_path`: **(Required)** Path to the fine-tuned UNet checkpoint.
* `--image_root_path`: **(Required)** Root folder containing input image sequences.
* `--mask_root_path`: **(Required)** Root folder containing input mask sequences.
* `--output_dir`: Directory to save all outputs. (Default: `output_batch`)

#### **Inference Configuration**
* `--num_frames`: Number of frames to generate. (Default: 16)
* `--num_input_frames`: Number of frames to read from input folders. (Default: same as `num_frames`)
* `--width`: Processing width for the frames. (Default: 1024)
* `--height`: Processing height for the frames. (Default: 576)
* `--keep_aspect_ratio`: If set, maintains the aspect ratio of the input images.
* `--mask_cond_mode`: Mask conditioning mode. (Choices: `vae`, `interpolate`, Default: `vae`)
* `--mixed_precision`: Use mixed precision for inference. (Choices: `no`, `fp16`, `bf16`, Default: `fp16`)
* `--seed`: A seed for reproducibility. (Default: 42)

#### **Mask Augmentation**
* `--mask_augmentation`: Type of augmentation to apply to the masks. (Choices: `none`, `polygon`, `downsample`, `bounding_box`, Default: `none`)
* `--downsample_factor`: Downsampling factor if `mask_augmentation` is `downsample`. (Default: 8)
* `--simplification_tolerance`: Simplification tolerance for `polygon` augmentation. (Default: 0.001)
* `--save_processed_mask`: If set, saves the final augmented masks that were fed into the model.

#### **Temporal Augmentation**
* `--temporal_augmentation`: If set, applies diverse temporal augmentations (occlusions, erosions, etc.) to random mask frames.
* `--num_occlusions`: Number of frames to apply temporal augmentation to. (Default: 1)
* `--occlusion_shape`: Shape for the temporal occlusion operation. (Choices: `rectangle`, `circle`, Default: `rectangle`)
* `--occlusion_scale_range`: Scale range for the occlusion, relative to the mask's bounding box. (Default: `[0.2, 0.5]`)
* `--erosion_dilation_kernel_size`: Kernel size for erosion and dilation operations. (Default: 5)

---

## Rocky Linux Local Setup

For a local Rocky Linux GPU machine, use the repository script below instead of `scripts/setup.sh`.
The original setup script assumes Ubuntu `apt`, Git LFS, and `conda`.

```bash
bash scripts/setup_rocky_inference.sh
source .venv/bin/activate
```

This creates `.venv`, installs inference-only Python dependencies, and downloads both required Hugging Face checkpoints into `checkpoints/` without Git LFS.

## Shared-Weights Layout (BCN production machines)

The BCN production Rocky boxes keep code, weights, and venvs on different
filesystems so the NFS mount holds the heavy shared artefacts and each machine
rebuilds its venvs locally on `/tmp`:

| Layer                        | Path                                                                |
|------------------------------|---------------------------------------------------------------------|
| Weights                      | `/mnt/production/project/bcn_lib/work/AI/VideoMaMa/checkpoints/`    |
| Code/config                  | `/mnt/production/user/<username>/DEV/AI/VideoMama/`                 |
| Inference venv (Python 3.9)  | `/tmp/videomama-venv/`                                              |
| SAM 3 UI venv (Python 3.12)  | `/tmp/videomama-sam3-ui-venv/`                                      |

### Bootstrap on any similar machine

```bash
cd /mnt/production/user/<username>/DEV/AI/VideoMama
bash scripts/bootstrap_tmp_venv.sh        # builds both venvs on /tmp
source .videomama-env                     # exports VIDEOMAMA_CHECKPOINTS etc.
source "$VIDEOMAMA_VENV/bin/activate"     # inference venv
```

Launch the production SAM 3 UI:

```bash
bash scripts/run_production_frame_ui.sh
```

Override any path at call time:

```bash
VIDEOMAMA_CHECKPOINTS=/some/other/path bash scripts/bootstrap_tmp_venv.sh
```

### Requirements

* Python 3.9 for the inference venv, Python 3.12 for the SAM 3 UI. `pyenv` is
  a convenient way to provide both.
* CUDA 12.4 capable GPU driver for the inference venv. SAM 3 follows Meta's
  current requirement of PyTorch 2.7+ and CUDA 12.6+; the bootstrap uses
  PyTorch 2.10.0 CUDA 12.8 wheels for the UI venv.
* Read access to `/mnt/production/project/bcn_lib/work/AI/VideoMaMa/` and
  write access to `/tmp`.

The CLI entry points read `VIDEOMAMA_CHECKPOINTS`, `VIDEOMAMA_BASE_MODEL_PATH`,
and `VIDEOMAMA_UNET_CHECKPOINT_PATH` from the environment, so you can redirect
them without editing code. SAM 3 checkpoints are resolved by the upstream SAM 3
package/Hugging Face access flow, so authenticate with `hf auth login` before
first SAM 3 use if the checkpoint is gated.

## EXR Input Support

`inference_onestep_folder.py` can read `.exr` frames directly for both image and mask sequences.
RGB EXR frames are converted to 8-bit RGB for the model, and mask EXR frames use a preferred channel via `--mask_channel`.

Useful flags:

* `--exr_gamma`: Display gamma used for RGB EXR conversion. Default: `2.2`
* `--exr_exposure`: Exposure offset in stops before gamma conversion. Default: `0.0`
* `--mask_channel`: Preferred EXR mask channel, such as `A`, `Y`, or `R`. Default: `A`

Example:

```bash
python inference_onestep_folder.py     --base_model_path checkpoints/stable-video-diffusion-img2vid-xt     --unet_checkpoint_path checkpoints/VideoMaMa     --image_root_path /path/to/image_exr_sequences     --mask_root_path /path/to/mask_sequences     --output_dir output_exr     --keep_aspect_ratio     --mask_channel A     --exr_gamma 2.2     --exr_exposure 0.0
```

## Output Structure

The script will generate an output directory with the following structure:

```
/path/to/your/output_dir/
├── mask_guide/ (if --save_processed_mask is used)
│   ├── video_001/
│   │   ├── frame_0000.png
│   │   └── ...
│   └── video_002/
│       └── ...
│
└── results/
    ├── video_001/
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   ├── ...
    │   └── video.mp4 (if more than one frame is generated)
    └── video_002/
        └── ...
```
