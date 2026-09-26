"""FlashVSR v1.1 4x video super-resolution for ComfyUI, tuned for RTX 4090.

One node with the native video UX — VIDEO in, VIDEO out — so it composes with
the core Load Video / Save Video nodes exactly like ComfyUI's built-in (cloud,
paid) FlashVSR node, except this one runs locally on your own GPU via the
flashvsr-sm89-ops operator pack (~1.3x over the official pipeline on a 4090,
no source edits, no CUDA build).

First execution: clones FlashVSR (if no checkout is found) and downloads the
v1.1 weights (~6.5 GB) into ComfyUI/models/flashvsr/. The heavy import chain
is deferred until the node runs.
"""
import contextlib
import os
import subprocess
import sys
import types
from fractions import Fraction

import numpy as np
import torch

from comfy_api.latest import ComfyExtension, InputImpl, io, Types

FLASHVSR_REPO = "https://github.com/OpenImagingLab/FlashVSR"
WEIGHTS_REPO = "JunhaoZhuang/FlashVSR-v1.1"
WEIGHT_FILES = (
    "config.json", "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt", "model_index.json", "TCDecoder.ckpt", "Wan2.1_VAE.pth",
)
# repo root (this file lives in comfyui/) — keeps the fallback lookup
# paths stable wherever the checkout is cloned
NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PIPE = None          # loaded FlashVSRTinyPipeline (cached across runs)


# --------------------------------------------------------------------------
# discovery / setup
# --------------------------------------------------------------------------

def _try_folder_paths():
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def _models_dir():
    fp = _try_folder_paths()
    if fp is not None:
        return os.path.join(fp.models_dir, "flashvsr")
    return os.path.join(os.path.dirname(NODE_DIR), "models", "flashvsr")


def _is_flashvsr(path):
    return path and os.path.isdir(os.path.join(path, "diffsynth"))


def ensure_flashvsr_checkout():
    """Locate (or clone) the FlashVSR repo whose diffsynth we import."""
    root = os.environ.get("FLASHVSR_ROOT")
    if _is_flashvsr(root):
        return root
    for cand in (os.path.join(_models_dir(), "FlashVSR"),
                 os.path.join(NODE_DIR, "FlashVSR")):
        if _is_flashvsr(cand):
            return cand
    target = os.path.join(NODE_DIR, "FlashVSR")
    print(f"[FlashVSR] cloning {FLASHVSR_REPO} -> {target} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", FLASHVSR_REPO, target],
                       check=True)
    except Exception as e:
        raise RuntimeError(
            "FlashVSR checkout not found. Set FLASHVSR_ROOT to your FlashVSR "
            "clone, or git clone it into ComfyUI/models/flashvsr/FlashVSR, or "
            f"install git so this node can clone it automatically. ({e})") from e
    return target


def ensure_weights(wanvsr=None):
    """Return a complete FlashVSR-v1.1 weight directory.

    Order: ComfyUI/models/flashvsr/FlashVSR-v1.1, then a copy that already
    lives inside the checkout's WanVSR dir (the quickstart layout), then
    download from HuggingFace into the models dir.
    """
    candidates = [os.path.join(_models_dir(), "FlashVSR-v1.1")]
    if wanvsr is not None:
        candidates.append(os.path.join(wanvsr, "FlashVSR-v1.1"))
    for wdir in candidates:
        if all(os.path.exists(os.path.join(wdir, f)) for f in WEIGHT_FILES):
            return wdir
    wdir = candidates[0]
    from huggingface_hub import snapshot_download
    print(f"[FlashVSR] downloading {len(WEIGHT_FILES)} weight file(s) from "
          f"{WEIGHTS_REPO} into {wdir} (~6.5 GB, once) ...")
    snapshot_download(WEIGHTS_REPO, local_dir=wdir)
    still = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(wdir, f))]
    if still:
        raise RuntimeError(f"weight download incomplete, missing: {still}")
    return wdir


# --------------------------------------------------------------------------
# pipeline loading (once; ~7 s + model load)
# --------------------------------------------------------------------------

def _load_pipeline(root, wanvsr, weights_dir):
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    sys.path.insert(0, root)
    sys.path.insert(0, wanvsr)

    # Order matters: the pack's import installs a zero-compile Triton stand-in
    # for block_sparse_attn when the CUDA package is absent.
    import flashvsr_sm89_ops  # noqa: F401

    # the official init path resolves EVERYTHING relative to the WanVSR dir —
    # "./FlashVSR-v1.1/..." for weights and "../../examples/WanVSR/prompt_tensor"
    # for the cached prompt KV — so sit in WanVSR for the duration. When the
    # weights live outside the checkout (ComfyUI models dir), link them in.
    target = os.path.join(wanvsr, "FlashVSR-v1.1")
    if os.path.abspath(target) != os.path.abspath(weights_dir) \
            and not all(os.path.exists(os.path.join(target, f)) for f in WEIGHT_FILES):
        try:
            if os.path.islink(target):
                os.remove(target)
            os.symlink(os.path.abspath(weights_dir), target)
        except OSError:
            pass  # a real dir already occupies the name; init will validate it
    cwd = os.getcwd()
    os.chdir(wanvsr)
    # the official entry imports "utils.utils"/"utils.TCDecoder" (WanVSR/utils,
    # a namespace dir). ComfyUI ships its own regular `utils` package which
    # already occupies sys.modules and would win the import; shadow it with a
    # synthetic package for the duration of the module exec, then restore.
    saved_utils = sys.modules.get("utils")
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [os.path.join(wanvsr, "utils")]
    utils_pkg.__package__ = "utils"
    sys.modules["utils"] = utils_pkg
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "infer_flashvsr_official", os.path.join(wanvsr, "infer_flashvsr_v1.1_tiny.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["infer_flashvsr_official"] = mod
        spec.loader.exec_module(mod)

        pipe = mod.init_pipeline()
    finally:
        if saved_utils is not None:
            sys.modules["utils"] = saved_utils
        else:
            sys.modules.pop("utils", None)
        os.chdir(cwd)
    applied = flashvsr_sm89_ops.enable(pipe)
    print(f"[FlashVSR] ops enabled: {applied}")
    print(f"[FlashVSR] pipeline ready (lcsa={flashvsr_sm89_ops.active_backend()})")
    _PIPE = pipe
    _release_gpu(pipe)
    return pipe


def _warm_gpu(pipe):
    """Bring the pipeline's weights onto the GPU for a run."""
    pipe.load_models_to_device(["dit", "vae"])
    pipe.TCDecoder.to("cuda")


def _release_gpu(pipe):
    """Park the pipeline in CPU RAM so other nodes get their VRAM back.

    load_models_to_device drives diffsynth's own cpu-offload for dit/vae;
    TCDecoder sits outside that registry and is parked by hand. The FP8
    linears installed by the ops pack have no offload hook and stay
    resident (~2 GB) — the full park/warm roundtrip is output-exact.
    """
    pipe.load_models_to_device([])
    pipe.TCDecoder.cpu()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def _streaming_progress(pipe, pbar):
    """Route the pipeline's streaming loop into ComfyUI's progress bar.

    The loop constructs tqdm(range(n)) directly, so swap the pipeline
    module's tqdm symbol for an adapter that feeds the ComfyUI bar.
    """
    mod = sys.modules[type(pipe).__module__]
    saved = mod.tqdm

    class _Bar:
        def __init__(self, iterable):
            self._it = iter(iterable)

        def __iter__(self):
            for i, item in enumerate(self._it):
                yield item
                pbar.update_absolute(i + 1)

        def update(self, n=1):
            pass

        def close(self):
            pass

    mod.tqdm = lambda iterable=None, *a, **k: _Bar(iterable or ())
    pbar.update_absolute(0)
    try:
        yield
    finally:
        mod.tqdm = saved


# --------------------------------------------------------------------------
# input preparation — mirrors the official example 1:1
# --------------------------------------------------------------------------

def _largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def _image_batch_to_lq(images, target_h=None, multiple=128):
    """ComfyUI IMAGE [B,H,W,C] float 0..1 -> (LQ [1,C,F,H,W] in [-1,1], th, tw, F).

    Same path as the official example: PIL bicubic x4, center-crop to a
    128-multiple, last frame repeated to reach the streaming minimum.
    With target_h, the input is pre-scaled so the 4x model lands near that
    output height (the official node's target_resolution contract).
    """
    from PIL import Image
    arr = (images.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    w0, h0 = arr.shape[2], arr.shape[1]

    # the model always runs 4x on its (bicubic-upsampled) input; with
    # target_h, choose the input size whose 4x lands nearest the requested
    # height and resize each original frame straight there (one interpolation)
    scale = 4.0
    if target_h is not None:
        s = target_h / h0
        w0 = max(1, int(round(w0 * s / 4.0)))
        h0 = max(1, int(round(h0 * s / 4.0)))

    tW = int(round(w0 * scale)) // multiple * multiple
    tH = int(round(h0 * scale)) // multiple * multiple
    if tW == 0 or tH == 0:
        raise ValueError(
            f"input {arr.shape[2]}x{arr.shape[1]} is too small: the 4x output "
            f"would fall below the {multiple}px minimum. Pick a larger "
            f"target_resolution.")

    idx = list(range(arr.shape[0])) + [arr.shape[0] - 1] * 4
    # the streaming loop needs (F-1)//8 >= 3, i.e. F >= 25; hold the last frame
    # on short clips (same trick as the official 4-frame pad, just more of it)
    MIN_PADDED = 25
    if len(idx) < MIN_PADDED:
        idx += [idx[-1]] * (MIN_PADDED - len(idx))
    F = _largest_8n1_leq(len(idx))
    idx = idx[:F]

    frames = []
    for i in idx:
        img = Image.fromarray(arr[i]).convert("RGB")
        sW, sH = int(round(w0 * scale)), int(round(h0 * scale))
        up = img.resize((sW, sH), Image.BICUBIC)
        l, t = (sW - tW) // 2, (sH - tH) // 2
        up = up.crop((l, t, l + tW, t + tH))
        t32 = torch.from_numpy(np.asarray(up, np.uint8)).float().permute(2, 0, 1)
        frames.append((t32 / 255.0 * 2.0 - 1.0).to(dtype=torch.bfloat16, device="cuda"))
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, tH, tW, F


# --------------------------------------------------------------------------
# the node — same shape as ComfyUI's built-in FlashVSR node, but local
# --------------------------------------------------------------------------

class FlashVSRUpscale(io.ComfyNode):
    """VIDEO in -> VIDEO out, on your own GPU."""

    # same vocabulary as the built-in WaveSpeed FlashVSR node
    _TARGET_HEIGHTS = {"720p": 720, "1080p": 1080, "2K": 1440, "4K": 2160}
    _SEED = 0  # the SR pipeline is visually seed-insensitive; keep it fixed

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSRUpscale",
            display_name="FlashVSR Video Upscale (sm89, local)",
            category="video/upscaler",
            description="Local, free counterpart of the built-in WaveSpeed "
            "FlashVSR node: FlashVSR v1.1 video super-resolution on your own "
            "NVIDIA GPU (no upload, no API key). Feed it from Load Video, "
            "save with Save Video.",
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip="The low-resolution clip. Very short clips are "
                    "padded by holding their last frame; audio is kept when "
                    "the frame count is unchanged."),
                io.Combo.Input(
                    "target_resolution",
                    options=list(cls._TARGET_HEIGHTS),
                    tooltip="Approximate output height. The output snaps to "
                    "the model's 128px grid. Peak VRAM for a ~3 s clip: "
                    "720p ≈ 9 GB, 1080p ≈ 19 GB, 2K ≈ 33 GB; 4K does not "
                    "fit a 24 GB card."),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, video: io.Video.Type, target_resolution: str) -> io.NodeOutput:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        components = video.get_components()
        images = components.images  # [F,H,W,C] float 0..1 (cpu)

        root = ensure_flashvsr_checkout()
        wanvsr = os.path.join(root, "examples", "WanVSR")
        weights_dir = ensure_weights(wanvsr)

        # ComfyUI-style memory handoff: let the model manager free whatever
        # upstream nodes left on the GPU before our pipeline comes in.
        mm.unload_all_models()
        pipe = _load_pipeline(root, wanvsr, weights_dir)
        _warm_gpu(pipe)

        LQ, th, tw, F = _image_batch_to_lq(
            images, target_h=cls._TARGET_HEIGHTS[target_resolution])
        print(f"[FlashVSR] {images.shape[0]} frames @ {images.shape[2]}x"
              f"{images.shape[1]} -> {tw}x{th}, running {F} frames")

        pbar = ProgressBar(max(1, (F - 1) // 8 - 2))  # the streaming loop's length
        try:
            with _streaming_progress(pipe, pbar):
                video_t = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0,
                    num_inference_steps=1, seed=cls._SEED, LQ_video=LQ,
                    num_frames=F, height=th, width=tw,
                    is_full_block=False, if_buffer=True,
                    topk_ratio=2.0 * 768 * 1280 / (th * tw),
                    kv_ratio=3.0, local_range=11, color_fix=True,
                )
        except torch.cuda.OutOfMemoryError:
            _release_gpu(pipe)
            raise RuntimeError(
                f"Out of GPU memory rendering {tw}x{th}. VRAM grows with "
                f"output pixels; for a ~3 s clip expect roughly 720p ≈ 9 GB, "
                f"1080p ≈ 19 GB, 2K ≈ 33 GB, and 4K does not fit even a "
                f"48 GB card. Pick a smaller target_resolution or a shorter "
                f"clip (Trim Video).") from None
        _release_gpu(pipe)
        # pipeline contract: [C,F,H,W] in [-1,1] -> ComfyUI IMAGE [F,H,W,C] 0..1
        out = ((video_t.float().clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 3, 0).cpu()
        del LQ
        torch.cuda.empty_cache()

        audio = components.audio if out.shape[0] == images.shape[0] else None
        fps = Fraction(components.frame_rate) if components.frame_rate else Fraction(30)
        out_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=out, audio=audio, frame_rate=fps))
        return io.NodeOutput(out_video)


class FlashVSRExtension(ComfyExtension):
    async def get_node_list(self):
        return [FlashVSRUpscale]


def comfy_entrypoint():
    return FlashVSRExtension()
