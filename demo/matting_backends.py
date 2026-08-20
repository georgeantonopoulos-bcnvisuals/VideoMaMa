"""
Matting backend registry for the production roto pipeline.

The production app can produce an alpha matte with more than one model. This
module is the single place that describes those backends: what they are called
in the UI, which checkpoint/config they use, what resolution they really run at,
and — critically — what identity a produced matte has so results from different
backends or settings can never be silently reused for each other.

Deliberately dependency-free (stdlib only). It is imported by the Gradio UI, by
the isolated SAM2Matting worker, and by tests, none of which should have to
agree on a torch version.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

# --- Upstream pinning -------------------------------------------------------
# SAM2Matting is research code without releases or tags. Pin an explicit commit
# so a rebuilt runtime reproduces the same numerics, and record the pin in every
# manifest. Refresh deliberately (SAM2MATTING_GIT_REF=... REFRESH_SAM2MATTING_LOCK=1).
SAM2MATTING_REPO_URL = "https://github.com/FudanCVL/SAM2Matting.git"
SAM2MATTING_PINNED_COMMIT = "73dd721d77b56749248aefe5e8824d7f61b9d13c"

# Hugging Face repo holding the three released checkpoints, pinned by revision.
SAM2MATTING_HF_REPO = "FudanCVL/SAM2Matting"
SAM2MATTING_HF_REVISION = "4315db9c60d27fde396b09765748a0ca6c97bed5"

# Runtimes live on a large local scratch disk, not /tmp: /tmp is the root
# filesystem on these hosts and cannot hold three CUDA venvs. Overridable with
# VIDEOMAMA_VENV_ROOT, or per-runtime with VIDEOMAMA_SAM2MATTING_VENV/_HOME.
DEFAULT_VENV_ROOT = "/mnt/temporal/VideoMama"
DEFAULT_SAM2MATTING_VENV = f"{DEFAULT_VENV_ROOT}/videomama-sam2matting-venv"
DEFAULT_SAM2MATTING_HOME = f"{DEFAULT_VENV_ROOT}/videomama-sam2matting-src"

RUNTIME_LOCK_RELPATH = Path("tmp") / "runtime-locks" / "sam2matting-runtime.json"


@dataclass(frozen=True)
class MattingBackendSpec:
    """Static description of one way to turn plate + SAM 3 mask into alpha."""

    backend_id: str
    label: str
    family: str  # 'videomama' | 'sam2matting'
    variant: str  # '' | 'sam2.1base+' | 'sam2.1tiny' | 'sam3'
    checkpoint_name: str  # file under <checkpoints>/SAM2Matting/, or '' for VideoMaMa
    config_name: str  # hydra config name inside the upstream sam2 package
    model_resolution: int  # square resolution the network really sees
    alpha_head_resolution: int  # resolution the alpha decoder emits before upsampling
    output_slug: str  # per-backend output directory suffix
    experimental: bool = False
    notes: str = ""

    def output_dirnames(self):
        """Return (matte_preview_dirname, alpha_dirname) for this backend.

        VideoMaMa keeps its historical directory names so existing runs on disk
        stay resumable; every other backend gets its own pair so two backends
        can coexist in one run root without overwriting each other.
        """
        if self.family == "videomama":
            return "videomama_frames", "alpha_frames"
        return f"matte_frames_{self.output_slug}", f"alpha_frames_{self.output_slug}"


VIDEOMAMA_BACKEND = MattingBackendSpec(
    backend_id="videomama",
    label="VideoMaMa (SVD generative matting)",
    family="videomama",
    variant="",
    checkpoint_name="",
    config_name="",
    model_resolution=0,  # driven by the UI processing resolution instead
    alpha_head_resolution=0,
    output_slug="videomama",
    notes=(
        "Generative SVD matting guided by a binary SAM mask. Slow, chunked, and "
        "seeded; useful as a rescue path and for comparison."
    ),
)

SAM2MATTING_BASE_PLUS = MattingBackendSpec(
    backend_id="sam2matting-sam2.1base+",
    label="SAM2Matting SAM2.1 Base+ (default)",
    family="sam2matting",
    variant="sam2.1base+",
    checkpoint_name="SAM2Matting-SAM2.1Base+.pt",
    config_name="configs/sam2matting-sam2.1base+.yaml",
    model_resolution=1024,
    alpha_head_resolution=512,
    output_slug="s2m_base_plus",
    notes=(
        "Preferred matte generator. Hiera-B+ VOS tracker plus a dedicated ROI/alpha "
        "head; predicts soft hair boundaries directly."
    ),
)

SAM2MATTING_TINY = MattingBackendSpec(
    backend_id="sam2matting-sam2.1tiny",
    label="SAM2Matting SAM2.1 Tiny (fast/experimental)",
    family="sam2matting",
    variant="sam2.1tiny",
    checkpoint_name="SAM2Matting-SAM2.1Tiny.pt",
    config_name="configs/sam2matting-sam2.1tiny.yaml",
    model_resolution=1024,
    alpha_head_resolution=512,
    output_slug="s2m_tiny",
    experimental=True,
    notes="Same heads on a Hiera-T tracker. Faster and lighter; weaker tracking.",
)

SAM2MATTING_SAM3 = MattingBackendSpec(
    backend_id="sam2matting-sam3",
    label="SAM2Matting SAM3 tracker (experimental)",
    family="sam2matting",
    variant="sam3",
    checkpoint_name="SAM2Matting-SAM3.pt",
    config_name="",  # built programmatically upstream, no hydra config
    model_resolution=1008,
    alpha_head_resolution=504,
    output_slug="s2m_sam3",
    experimental=True,
    notes=(
        "Uses SAM2Matting's own vendored SAM 3 tracker at 1008px. Independent of "
        "our official SAM 3 install and not validated in production here."
    ),
)

_BACKENDS = (
    SAM2MATTING_BASE_PLUS,
    VIDEOMAMA_BACKEND,
    SAM2MATTING_TINY,
    SAM2MATTING_SAM3,
)

DEFAULT_BACKEND_ID = SAM2MATTING_BASE_PLUS.backend_id
_BY_ID = {spec.backend_id: spec for spec in _BACKENDS}
_BY_LABEL = {spec.label: spec for spec in _BACKENDS}


def backend_labels():
    """UI dropdown choices, preferred backend first."""
    return [spec.label for spec in _BACKENDS]


def resolve_backend(value):
    """Accept a backend id, a UI label, or a spec and return the spec."""
    if isinstance(value, MattingBackendSpec):
        return value
    key = str(value or DEFAULT_BACKEND_ID)
    if key in _BY_ID:
        return _BY_ID[key]
    if key in _BY_LABEL:
        return _BY_LABEL[key]
    raise ValueError(f"Unknown matting backend: {value!r}")


def is_sam2matting(value):
    return resolve_backend(value).family == "sam2matting"


# --- Conditioning strategy --------------------------------------------------
# SAM2Matting conditions on masks at arbitrary frames. We already have both
# artist keyframes and a full SAM 3 mask set, so the question is only how often
# to re-anchor the tracker on SAM 3 rather than let it drift on its own memory.
CONDITIONING_STRATEGIES = {
    "Keyframes + periodic SAM 3 (recommended)": "periodic",
    "Artist keyframes only": "keyframes",
    "Dense SAM 3 guidance (every frame)": "dense",
}
DEFAULT_CONDITIONING_LABEL = "Keyframes + periodic SAM 3 (recommended)"
# Measured on a 4K uni_island plate (thin branches, hard tracking): conditioning
# only on artist keyframes lost the object entirely by frame 2, while a 4-frame
# re-anchor held it with one QC-flagged frame and a 2-frame re-anchor held it
# cleanly. Anchoring is nearly free (23.6s vs 22.2s over 12 frames), so the
# default errs towards frequent re-anchoring on the SAM 3 masks we trust.
DEFAULT_GUIDANCE_INTERVAL = 4


def resolve_conditioning(value):
    key = str(value or DEFAULT_CONDITIONING_LABEL)
    if key in CONDITIONING_STRATEGIES:
        return CONDITIONING_STRATEGIES[key]
    if key in set(CONDITIONING_STRATEGIES.values()):
        return key
    raise ValueError(f"Unknown conditioning strategy: {value!r}")


def conditioning_frames(strategy, range_start, range_end, keyframe_indices,
                        guidance_interval=DEFAULT_GUIDANCE_INTERVAL):
    """Choose which frames feed a SAM 3 mask into SAM2Matting as conditioning.

    Every strategy always conditions on the first frame of the processed range:
    SAM2Matting propagates from the earliest conditioning frame, so without it a
    range starting mid-shot would have nothing to anchor on. Artist keyframes
    inside the range are always included because they are the corrections the
    artist actually paid for.
    """
    strategy = resolve_conditioning(strategy)
    range_start = int(range_start)
    range_end = int(range_end)
    if range_end < range_start:
        raise ValueError(f"Empty conditioning range [{range_start}, {range_end}].")

    if strategy == "dense":
        return list(range(range_start, range_end + 1))

    frames = {range_start}
    frames.update(
        int(idx) for idx in (keyframe_indices or ())
        if range_start <= int(idx) <= range_end
    )
    if strategy == "periodic":
        interval = max(1, int(guidance_interval))
        frames.update(range(range_start, range_end + 1, interval))
        # Anchor the tail too, so the last stretch is never unguided.
        frames.add(range_end)
    return sorted(frames)


# --- Runtime discovery ------------------------------------------------------

def _venv_root():
    return os.environ.get("VIDEOMAMA_VENV_ROOT") or DEFAULT_VENV_ROOT


def runtime_lock_path(repo_root):
    return Path(repo_root) / RUNTIME_LOCK_RELPATH


def read_runtime_lock(repo_root):
    """Read what scripts/bootstrap_sam2matting.sh recorded about the runtime.

    Returns an empty dict when SAM2Matting has never been bootstrapped, so the
    caller can raise a helpful error instead of failing on a missing file.
    """
    path = runtime_lock_path(repo_root)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sam2matting_python(repo_root):
    """Resolve the interpreter of the isolated SAM2Matting venv."""
    lock = read_runtime_lock(repo_root)
    venv = (
        os.environ.get("VIDEOMAMA_SAM2MATTING_VENV")
        or lock.get("venv")
        or f"{_venv_root()}/videomama-sam2matting-venv"
    )
    return Path(venv) / "bin" / "python"


def sam2matting_home(repo_root):
    """Resolve the pinned SAM2Matting source checkout."""
    lock = read_runtime_lock(repo_root)
    home = (
        os.environ.get("VIDEOMAMA_SAM2MATTING_HOME")
        or lock.get("home")
        or f"{_venv_root()}/videomama-sam2matting-src"
    )
    return Path(home)


def sam2matting_checkpoint_dir(checkpoints_root):
    return Path(checkpoints_root) / "SAM2Matting"


def file_digest(path, chunk_size=1 << 20):
    """Hash a checkpoint so a swapped weights file invalidates cached mattes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Parameters and identity ------------------------------------------------

@dataclass
class SAM2MattingParams:
    """Everything that changes SAM2Matting's numeric output."""

    conditioning: str = "periodic"
    guidance_interval: int = DEFAULT_GUIDANCE_INTERVAL
    frame_long_edge_cap: int = 2048
    window_size: int = 300
    window_overlap: int = 8
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    bf16: bool = True
    mask_threshold: int = 127

    def normalized(self):
        return SAM2MattingParams(
            conditioning=resolve_conditioning(self.conditioning),
            guidance_interval=max(1, int(self.guidance_interval)),
            frame_long_edge_cap=max(256, int(self.frame_long_edge_cap)),
            window_size=max(0, int(self.window_size)),
            window_overlap=max(0, int(self.window_overlap)),
            offload_video_to_cpu=bool(self.offload_video_to_cpu),
            offload_state_to_cpu=bool(self.offload_state_to_cpu),
            bf16=bool(self.bf16),
            mask_threshold=max(0, min(int(self.mask_threshold), 254)),
        )

    def as_dict(self):
        return asdict(self.normalized())


@dataclass
class VideoMaMaParams:
    """Everything that changes VideoMaMa's numeric output."""

    mask_cond_mode: str = "vae"
    seed: int = 42
    fps: int = 7
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.0
    guide_expand_px: int = 0
    guide_source: str = "sam3"
    guide_threshold: float = 0.5
    guide_backend_id: str = ""
    guide_settings_hash: str = ""
    chunk_size: int = 16
    overlap: int = 4
    refine_edges_against_plate: bool = False
    processing_size: tuple = (1024, 576)

    def as_dict(self):
        payload = asdict(self)
        payload["processing_size"] = [int(v) for v in self.processing_size]
        return payload


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def backend_identity(spec, params, runtime=None, checkpoint_digest=None):
    """Identity of *the model*, independent of the shot being processed.

    Folded into every matte's settings hash so a matte produced by Base+ can
    never be mistaken for one produced by Tiny, VideoMaMa, or the same backend
    at a different upstream revision.
    """
    spec = resolve_backend(spec)
    identity = {
        "backend_id": spec.backend_id,
        "family": spec.family,
        "variant": spec.variant,
        "model_resolution": spec.model_resolution,
        "params": params.as_dict() if hasattr(params, "as_dict") else dict(params or {}),
    }
    if spec.family == "sam2matting":
        runtime = dict(runtime or {})
        identity.update({
            "upstream_repo": SAM2MATTING_REPO_URL,
            "upstream_commit": runtime.get("commit") or SAM2MATTING_PINNED_COMMIT,
            "checkpoint_name": spec.checkpoint_name,
            "checkpoint_revision": runtime.get("checkpoint_revision") or SAM2MATTING_HF_REVISION,
            "checkpoint_sha256": checkpoint_digest or runtime.get(
                "checkpoint_sha256", {}
            ).get(spec.checkpoint_name, ""),
            "torch_version": runtime.get("torch_version", ""),
        })
    else:
        identity.update({
            "base_model_path": os.environ.get("VIDEOMAMA_BASE_MODEL_PATH", ""),
            "unet_checkpoint_path": os.environ.get("VIDEOMAMA_UNET_CHECKPOINT_PATH", ""),
        })
    return identity
