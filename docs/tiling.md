# Spatial tiling: 4K on a 24 GB card

`render_tiled` renders an output canvas larger than VRAM allows by running the
**identical pipeline call once per overlapping spatial tile** and blending the
tiles back together. It is what the ComfyUI node uses automatically above
~2.2 MP, and it is available to direct callers of the pack.

## Why

Measured VRAM for the FlashVSR pipeline (85-frame clip, weights included):

```
peak ≈ 1.9 GiB fixed + 8.6 GiB per output megapixel
```

A 4K canvas (3840×2048 = 7.9 MP) asks for ~44.8 GiB — an instant OOM on a
24 GB card. A tile of ≤ 2.1 MP (the official 1080p workload's shape) peaks at
~20 GiB. Measured (`benchmarks/tiled_render.json`, RTX 4090 D, 85 frames):

| canvas | untiled | tiled | wall |
|---|---|---|---|
| 2K 2560×1408 | 32.8 GiB | **20.7 GiB** (2 strips) | 20.4 s → 22.8 s (+12%) |
| 4K 3840×2048 | OOM at 44.8 GiB | **19.96 GiB** (5 strips 1024×2048) | 60.9 s |

Tiling is a memory technique only. The sm89 operators live in the DiT modules,
so every tile runs the same fused kernels; the A/B in
`benchmarks/tiled_ops_ab.json` measures **1.24×** with ops on vs off on the
tiled path (same as the untiled 1.17–1.30× within drift), and FP8 shaves a
further ~1.3 GiB off the tiled peak.

## The planner (`plan_tiles`)

```python
from flashvsr_sm89_ops import plan_tiles
tiles = plan_tiles(W, H)   # -> [(x, y, w, h), ...]
```

- **Full frame below `TILE_THRESHOLD` = 2.2 MP.** 1080p and below never tile.
- Otherwise **1D strips first** (vertical, then horizontal), 2 to 8 pieces —
  one seam direction only. A 2D grid (2–6 × 2–6, fewest seams then fewest
  pieces) is the fallback for canvases no strip count can bound.
- Everything sits on the pipeline's 128-px output grid (`MULT`). Strip width
  is snapped up from `(canvas + (n-1)·overlap + n-1) / n`, which guarantees
  stride ≤ piece − overlap, i.e. shared spans of at least `OVERLAP` px.
- `starts[-1] = canvas - piece` absorbs the floor-division remainder, so the
  last tile always reaches the edge.

## Blending (`tile_weight`)

Neighbours share ≥ 256 px (`OVERLAP` ≥ 2·MULT, so seams never touch the
window-quantised interior). Each tile carries per-axis linear ramps
(0→1 rising into its left/top neighbour, 1→0 falling into its right/bottom),
chosen so **adjacent ramps sum to exactly 1** across every overlap — a
partition of unity, verified offline to atol 1e-5 over canvas sweeps. The
blended result is bit-faithful to the tiles; nothing is double-drawn.

## Sizing to the card (`auto_tile_limits`)

The defaults above assume a 24 GB card. `auto_tile_limits(num_frames=F)`
probes the VRAM that is actually free (`torch.cuda.mem_get_info` after
`empty_cache`) and shrinks both constants to fit, so smaller cards tile more
aggressively instead of OOMing:

- target scales with the measured VRAM model — `8.6 GiB/MP × F/85` of free
  budget (minus a 1.25 GiB safety margin), clamped to at most the defaults
  and floored at 0.64 MP;
- the threshold follows the target, so a full-frame canvas the card cannot
  afford (e.g. 1080p ≈ 19 GB on a 16 GB card) is tiled too — measured case:
  16 GB card → 1080p as two 1152×1024 strips at ~1.2 MP;
- a roomy card gets the defaults unchanged, and any caller without CUDA gets
  the defaults blindly — the probe never *raises*.

The ComfyUI node pairs this with need-based eviction: it checks free VRAM
first and only calls `unload_all_models()` when the card could not offer the
~22 GiB budget any bucket needs — on a roomy card upstream models stay
resident and their next run is instant.

```python
from flashvsr_sm89_ops import auto_tile_limits

threshold, target = auto_tile_limits(num_frames=F)
tiles = plan_tiles(W, H, target=target, threshold=threshold)
```

## Per-tile pipeline call

Each tile is the *same* `pipe(...)` call you would issue for an
official-shape canvas, with three rewrites:

1. `LQ_video` — the tile's crop of the caller's bicubic-×4 canvas,
2. `height` / `width` — the tile size,
3. `topk_ratio` — rescaled by `canvas_area / tile_area`, so the sparse
   attention token budget lands on the same operating point as the official
   workload regardless of tile shape.

`num_frames` defaults to the canvas's own frame count. The accumulator is
fp32 on the host and the LQ canvas stays in CPU RAM — only one tile's crop is
ever resident on the GPU, which is the whole point at 4K. Returns
`[C, F, H, W]`, the pipeline's own output rank (4D or 5D input accepted).

```python
from flashvsr_sm89_ops import render_tiled

video = render_tiled(
    pipe, LQ_video,                 # [C,F,H,W] or [1,C,F,H,W], [-1,1], bicubic-×4 canvas
    num_frames=F, height=H, width=W,
    # ...any other pipe(...) kwargs, passed through verbatim
)
```

Knobs: `tile_threshold` / `tile_target` / `overlap` mirror the module
constants. `pipe` is duck-typed — anything callable like the FlashVSR
diffsynth pipeline works.

## Quality evidence

`benchmarks/tiled_render.json` (2K full vs 2K tiled, same clip):

- tiled-vs-full PSNR 31.6 dB — the two runs see different per-tile noise
  realizations (the SR pipeline is generative); this number establishes
  "same scene, same detail", not bitwise equality;
- seam-column error profile peaks at 0.0187, *below* the frame's global
  column-error maximum of 0.0270 — no seam signature;
- `benchmarks/tiled_render_seam.png` is the contact sheet (middle frame,
  ±320 px around the seam, tiled over full).

End-to-end through the live ComfyUI node: 4K job succeeds (ffprobe
3840×2048, 85 frames), progress events stream continuously across tiles on
one bar (2K → 18/18, 4K → 45 blocks), and `comfyui/selftest.py` stays green.

## Reproduce

```bash
# offline tile-math suite (CPU, no GPU needed):
python comfyui/test_tiling.py

# measured evidence — 2K full / 2K tiled / 4K tiled over one clip:
COMFYUI_ROOT=/path/to/ComfyUI python comfyui/bench_tiled.py <clip.mp4>
# -> benchmarks/tiled_render.json

# ops-on vs ops-off A/B on the tiled path (separate processes; flags are
# read at import time):
python comfyui/bench_tiled.py <clip.mp4>                       # ops on (default)
FS89_FP8= FS89_FUSED_ROPE=0 FS89_FUSED_ADALN=0 \
    python comfyui/bench_tiled.py <clip.mp4>                   # ops off
```

## Limits

- Wall-time cost ~+12% at 2K (two tiles); more tiles at 4K scale accordingly.
- VRAM still scales with clip length — quoted peaks are for ~3 s / 85-frame
  clips; trim longer clips (ComfyUI *Trim Video*) or accept higher peaks.
- The planner raises `ValueError` rather than emitting a tile smaller than
  `overlap + MULT` px; no canvas up to the tested sweeps hits this.
