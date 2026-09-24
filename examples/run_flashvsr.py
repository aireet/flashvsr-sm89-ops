#!/usr/bin/env python3
"""Run FlashVSR v1.1 Tiny on an RTX 4090 with the sm89-ops pack — no patching.

This mirrors the official example (examples/WanVSR/infer_flashvsr_v1.1_tiny.py
in the FlashVSR repo) end to end: same input preparation, same pipeline call,
same output naming. The only difference is one call — ``enable(pipe)`` — which
wires in FP8 linears, the fused bf16 kernels, TCDecoder channels_last and the
zero-compile Triton attention backend.

Quickstart (see docs/benchmarks.md to reproduce the timing numbers):

    git clone https://github.com/OpenImagingLab/FlashVSR
    python run_flashvsr.py --flashvsr-root /path/to/FlashVSR \
        --input /path/to/video.mp4          # or a directory of frames

Weights auto-download from HuggingFace (JunhaoZhuang/FlashVSR-v1.1, ~6.5 GB)
on first run. Run from any directory; results land in --out-dir.

Compare against the stock pipeline with --no-ops (same process, same weights).
"""
import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flashvsr_sm89_ops  # noqa: F401  (installs the Triton BSA stub — keep first)

REPO_ID = "JunhaoZhuang/FlashVSR-v1.1"
WEIGHT_FILES = [
    "config.json", "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt", "model_index.json", "TCDecoder.ckpt", "Wan2.1_VAE.pth",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", required=True,
                   help="input video file or frame directory (repeatable)")
    p.add_argument("--flashvsr-root", default=os.environ.get("FLASHVSR_ROOT"),
                   help="FlashVSR checkout (default: $FLASHVSR_ROOT)")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--sparse-ratio", type=float, default=2.0,
                   help="1.5 -> faster, 2.0 -> more stable (official guidance)")
    p.add_argument("--local-range", type=int, default=11,
                   help="9 -> sharper details, 11 -> more stable (official guidance)")
    p.add_argument("--no-ops", action="store_true",
                   help="disable this pack; run the stock pipeline for A/B")
    p.add_argument("--no-download", action="store_true",
                   help="fail instead of downloading missing weights")
    return p.parse_args()


def ensure_weights(wanvsr_dir, download=True):
    wdir = os.path.join(wanvsr_dir, "FlashVSR-v1.1")
    missing = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(wdir, f))]
    if not missing:
        return wdir
    if not download:
        raise FileNotFoundError(f"missing weights in {wdir}: {missing}")
    from huggingface_hub import snapshot_download
    print(f"[run_flashvsr] downloading {missing} from {REPO_ID} (~6.5 GB total) ...")
    snapshot_download(REPO_ID, local_dir=wdir)
    still = [f for f in WEIGHT_FILES if not os.path.exists(os.path.join(wdir, f))]
    if still:
        raise FileNotFoundError(f"download incomplete, missing: {still}")
    return wdir


def load_official_example(wanvsr_dir):
    """Import the official entry module so input prep and saving match it 1:1."""
    path = os.path.join(wanvsr_dir, "infer_flashvsr_v1.1_tiny.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"official example not found at {path} — "
                                f"is --flashvsr-root a FlashVSR checkout?")
    spec = importlib.util.spec_from_file_location("infer_flashvsr_official", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infer_flashvsr_official"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    args = parse_args()
    if not args.flashvsr_root:
        sys.exit("error: pass --flashvsr-root or set $FLASHVSR_ROOT to a FlashVSR checkout")

    root = os.path.abspath(args.flashvsr_root)
    wanvsr = os.path.join(root, "examples", "WanVSR")
    if not os.path.isdir(os.path.join(root, "diffsynth")):
        sys.exit(f"error: {root} does not look like a FlashVSR checkout (no diffsynth/)")

    ensure_weights(wanvsr, download=not args.no_download)

    # The official example resolves models/results relative to its own directory.
    out_dir = os.path.abspath(args.out_dir)
    os.chdir(wanvsr)
    sys.path.insert(0, root)
    sys.path.insert(0, wanvsr)

    import torch
    official = load_official_example(wanvsr)

    t0 = time.time()
    pipe = official.init_pipeline()
    if not args.no_ops:
        flashvsr_sm89_ops.enable(pipe)
    print(f"[run_flashvsr] pipeline ready in {time.time() - t0:.1f}s "
          f"(lcsa={flashvsr_sm89_ops.active_backend()})")

    os.makedirs(out_dir, exist_ok=True)
    for p in args.input:
        p = os.path.abspath(p)
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip("/"))
        t0 = time.time()
        try:
            LQ, th, tw, F, fps = official.prepare_input_tensor(
                p, scale=args.scale, dtype=torch.bfloat16, device="cuda")
        except Exception as e:
            print(f"[Error] {name}: {e}")
            continue
        if F < 25:
            # the streaming loop needs (F-1)//8 >= 3; hold the last frame
            reps = 25 - F
            LQ = torch.cat([LQ, LQ[:, :, -1:].expand(-1, -1, reps, -1, -1)], dim=2)
            F = 25
            print(f"[run_flashvsr] {name}: {reps} frame(s) held to reach the "
                  f"streaming minimum (F=25)")

        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
            seed=args.seed, LQ_video=LQ, num_frames=F, height=th, width=tw,
            is_full_block=False, if_buffer=True,
            topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=3.0, local_range=args.local_range, color_fix=True,
        )
        gen_ms = (time.time() - t0) * 1000
        frames = official.tensor2video(video)
        save_path = os.path.join(
            out_dir, f"FlashVSR_v1.1_Tiny_{name.split('.')[0]}_seed{args.seed}.mp4")
        official.save_video(frames, save_path, fps=fps, quality=6)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[run_flashvsr] {name}: {gen_ms:.0f} ms, {F - 4} frames, "
              f"{(F - 4) / gen_ms * 1000:.2f} FPS, peak {peak:.1f} GB -> {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
