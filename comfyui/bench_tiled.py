#!/usr/bin/env python3
"""Full-frame vs tiled render evidence: VRAM peaks, wall time, quality.

Runs the node's own pipeline three times over one clip — 2K full-frame,
2K tiled, 4K tiled — and writes benchmarks/tiled_render.json. GPU and
weights required.

    COMFYUI_ROOT=/root/ComfyUI python comfyui/bench_tiled.py <clip.mp4>
"""
import json
import os
import sys
import time
import types
from fractions import Fraction

sys.path.insert(0, os.environ.get("COMFYUI_ROOT", "/root/ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.modules["folder_paths"] = types.SimpleNamespace(
    models_dir=os.path.join(os.environ.get("COMFYUI_ROOT", "/root/ComfyUI"), "models"),
    get_output_directory=lambda: os.environ.get("SELFTEST_OUT", "/tmp/comfy_selftest_out"),
)

import comfyui.nodes as node_mod  # noqa: E402
from comfy_api.latest import InputImpl, Types  # noqa: E402
from flashvsr_sm89_ops.tiling import plan_tiles, render_tiled  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmarks")
OUT_JSON = os.path.join(OUT_DIR, "tiled_render.json")
PIPE_KW = dict(prompt="", negative_prompt="", cfg_scale=1.0,
               num_inference_steps=1, seed=0, is_full_block=False,
               if_buffer=True, kv_ratio=3.0, local_range=11, color_fix=True)


def psnr(a, b):
    mse = (a.double() - b.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def run(pipe, LQ, F, topk, tiled=True):
    kw = dict(PIPE_KW, topk_ratio=topk, num_frames=F)
    if not tiled:
        kw["tile_threshold"] = float("inf")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = render_tiled(pipe, LQ, **kw)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    return out, dt, peak


def norm01(t):
    """pipeline output ([C,F,H,W] or [1,...]), range [-1,1] -> [C,F,H,W] 0..1"""
    t = t[0] if t.dim() == 5 else t
    return ((t.float().clamp(-1, 1) + 1) / 2).cpu()


def main():
    src = sys.argv[1]
    r = imageio.get_reader(src)
    frames = np.stack([f for f in r]).astype(np.float32) / 255.0
    fps = Fraction(round(r.get_meta_data().get("fps", 30)))
    r.close()
    images = torch.from_numpy(frames)
    print(f"[bench] input {tuple(images.shape)} @ {float(fps):g} fps")

    root = node_mod.ensure_flashvsr_checkout()
    wanvsr = os.path.join(root, "examples", "WanVSR")
    pipe = node_mod._load_pipeline(root, wanvsr,
                                   node_mod.ensure_weights(wanvsr))
    node_mod._warm_gpu(pipe)

    def make_lq(target_h):
        LQ, th, tw, F = node_mod._image_batch_to_lq(images, target_h=target_h)
        return LQ, th, tw, F

    results = {"input": os.path.basename(src),
               "input_frames": int(images.shape[0]),
               "gpu": torch.cuda.get_device_name(0)}

    # 2K full-frame vs tiled — the quality + VRAM comparison
    LQ, th, tw, F = make_lq(1440)
    topk2k = 2.0 * 768 * 1280 / (th * tw)
    full, dt_f, peak_f = run(pipe, LQ, F, topk2k, tiled=False)
    print(f"[bench] 2K full : {dt_f:6.1f}s  peak {peak_f:5.2f} GiB")
    tiled, dt_t, peak_t = run(pipe, LQ, F, topk2k, tiled=True)
    print(f"[bench] 2K tiled: {dt_t:6.1f}s  peak {peak_t:5.2f} GiB  "
          f"tiles {len(plan_tiles(tw, th))}")
    a, b = norm01(full), norm01(tiled)
    p = psnr(a, b)

    # seam check: no per-column error spike where tiles meet
    col = (a - b).abs().mean(dim=(0, 1, 2))  # [W]
    seam_x = [t[0] for t in plan_tiles(tw, th)[1:]]
    zones = [(max(0, x - 64), min(tw, x + 320)) for x in seam_x]
    seam_peak = max(float(col[z0:z1].max()) for z0, z1 in zones)
    global_peak = float(col.max())
    # visual evidence: middle frame around the seam, tiled on top, full below
    from PIL import Image
    fi = a.shape[1] // 2
    fa = a[:, fi].permute(1, 2, 0).numpy()
    fb = b[:, fi].permute(1, 2, 0).numpy()
    sx = seam_x[0]
    za, zb = fa[:, sx - 320:sx + 320], fb[:, sx - 320:sx + 320]
    strip = np.concatenate([za, np.full((12, za.shape[1], 3), 0.5), zb], axis=0)
    seam_png = os.path.join(OUT_DIR, "tiled_render_seam.png")
    Image.fromarray((strip.clip(0, 1) * 255).astype(np.uint8)).save(seam_png)

    results["2k"] = {
        "canvas": [tw, th], "tiles_full": 1,
        "tiles_tiled": len(plan_tiles(tw, th)),
        "psnr_tiled_vs_full_db": round(p, 3),
        "seam_col_err": round(seam_peak, 5),
        "global_col_err_max": round(global_peak, 5),
        "full_s": round(dt_f, 1), "full_peak_gib": round(peak_f, 2),
        "tiled_s": round(dt_t, 1), "tiled_peak_gib": round(peak_t, 2),
        "seam_png": "tiled_render_seam.png",
    }
    print(f"[bench] 2K PSNR(tiled vs full) {p:.2f} dB | seam col err "
          f"{seam_peak:.4f} vs global max {global_peak:.4f}")

    del LQ, full, tiled, a, b
    torch.cuda.empty_cache()

    # 4K tiled — the run that cannot exist untiled
    LQ, th, tw, F = make_lq(2160)
    tiles4k = plan_tiles(tw, th)
    o5, dt_4, peak_4 = run(pipe, LQ, F, 2.0 * 768 * 1280 / (th * tw), tiled=True)
    o = norm01(o5)
    assert not torch.isnan(o).any(), "NaN in 4K output"
    results["4k"] = {
        "canvas": [tw, th], "tiles": len(tiles4k),
        "tile_size": [tiles4k[0][2], tiles4k[0][3]],
        "s": round(dt_4, 1), "peak_gib": round(peak_4, 2),
        "frames": int(o.shape[1]),
        "range": [round(float(o.min()), 4), round(float(o.max()), 4)],
    }
    print(f"[bench] 4K tiled: {dt_4:.1f}s  peak {peak_4:.2f} GiB  "
          f"({len(tiles4k)} tiles of {tiles4k[0][2]}x{tiles4k[0][3]})")

    node_mod._release_gpu(pipe)
    results["resident_after_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
