"""Spatially tiled rendering: the same output canvas, bounded VRAM.

FlashVSR renders the whole output canvas in one pipeline call — VRAM grows
with output pixels (measured ~1.9 GiB fixed + ~8.6 GiB per output megapixel
for an 85-frame clip), so 4K does not fit a 24 GB card untiled. render_tiled
runs the identical pipeline call once per overlapping spatial tile, each at
most TILE_TARGET pixels so every call looks like the official 1080p workload
to the model, and blends the tiles back with linear ramps across the shared
256 px overlap. Quality versus the untiled render is frozen in
benchmarks/tiled_render.json.

The pipeline is duck-typed: anything callable like the FlashVSR diffsynth
pipeline works. Blending happens on the host, so no more than one tile ever
lives on the GPU.
"""

import torch

# the pipeline's output grid (px) — tiles and overlaps sit on it
MULT = 128
# shared pixels between neighbouring tiles; >= 2*MULT so seams never touch
# the window-quantised interior
OVERLAP = 256
# above this canvas size (px) we tile; 1080p full-frame (1.97 MP, ~19 GiB)
# stays on the untiled path
TILE_THRESHOLD = 2_200_000
# per-tile pixel budget: ~1.9 GiB fixed + 8.6 GiB/MP -> ~20 GiB peak, which
# fits a 24 GB card with headroom
TILE_TARGET = 2_100_000


def _snap_up(v, mult=MULT):
    return -(-v // mult) * mult


def _strip_starts(canvas, piece, n):
    """Starts of n pieces of `piece` px covering `canvas` on one axis."""
    stride = (canvas - piece) // (n - 1)
    # piece >= (canvas + (n-1)*OVERLAP)/n keeps stride <= piece - OVERLAP,
    # so shared spans stay at least OVERLAP wide; the exact last start absorbs
    # the floor-division remainder and guarantees edge coverage
    starts = [i * stride for i in range(n)]
    starts[-1] = canvas - piece
    return starts


def plan_tiles(W, H, target=TILE_TARGET, threshold=TILE_THRESHOLD,
               overlap=OVERLAP, mult=MULT):
    """Cover a W x H canvas with overlapping tiles on the `mult` grid.

    Prefers the full frame, then 1D strips (one seam direction, the shape
    production deployments validated), a 2D grid only as the fallback for
    canvases no strip count can bound. Returns [(x, y, w, h), ...].
    """
    if W * H <= threshold:
        return [(0, 0, W, H)]
    min_piece = overlap + mult

    def piece_fits(canvas, cross, n):
        piece = _snap_up((canvas + (n - 1) * overlap + n - 1) // n)
        return piece, (piece * cross <= target and min_piece <= piece <= canvas)

    for n in range(2, 9):  # vertical strips, full height
        piece, ok = piece_fits(W, H, n)
        if ok:
            return [(x, 0, piece, H) for x in _strip_starts(W, piece, n)]
    for n in range(2, 9):  # horizontal strips, full width
        piece, ok = piece_fits(H, W, n)
        if ok:
            return [(0, y, W, piece) for y in _strip_starts(H, piece, n)]

    # 2D fallback: fewest seams first, then fewest pieces
    best = None
    for cols in range(2, 7):
        for rows in range(2, 7):
            w = _snap_up((W + (cols - 1) * overlap + cols - 1) // cols)
            h = _snap_up((H + (rows - 1) * overlap + rows - 1) // rows)
            if w * h > target or w < min_piece or h < min_piece:
                continue
            key = (cols + rows - 2, cols * rows)
            if best is None or key < best[0]:
                best = (key, w, h, cols, rows)
    if best is None:
        raise ValueError(f"cannot tile {W}x{H} into ~{target} px pieces")
    _, w, h, cols, rows = best
    return [(x, y, w, h)
            for y in _strip_starts(H, h, rows)
            for x in _strip_starts(W, w, cols)]


def _axis_weights(starts, piece):
    """Per-piece weight vectors on one axis: ramps sum to 1 across overlaps."""
    out = []
    for i, s in enumerate(starts):
        w = torch.ones(piece, dtype=torch.float32)
        if i > 0:  # rise across the span shared with the left/top neighbour
            ov = starts[i - 1] + piece - s
            w[:ov] = torch.linspace(0.0, 1.0, ov + 2)[1:-1]
        if i < len(starts) - 1:  # fall across the right/bottom neighbour's
            ov = s + piece - starts[i + 1]
            w[-ov:] = torch.linspace(1.0, 0.0, ov + 2)[1:-1]
        out.append(w)
    return out


def tile_weight(x, y, tiles):
    """Blending weight of one tile as a [h, w] map; all tiles sum to 1."""
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    wx = _axis_weights(xs, tiles[0][2])[xs.index(x)]
    wy = _axis_weights(ys, tiles[0][3])[ys.index(y)]
    return wy[:, None] * wx[None, :]


def render_tiled(pipe, lq_video, *, overlap=OVERLAP, tile_target=TILE_TARGET,
                 tile_threshold=TILE_THRESHOLD, **pipe_kwargs):
    """Render lq_video [C,F,H,W] through `pipe` one spatial tile at a time.

    lq_video is the caller's bicubic-x4 canvas in [-1, 1], [C,F,H,W] or
    [1,C,F,H,W]. Keep it in CPU RAM: only one tile's crop is resident on
    the GPU at a time, which is the whole point at 4K. Every tile runs the
    exact pipeline call requested via pipe_kwargs, with height, width,
    LQ_video and topk_ratio rewritten per tile (the token budget scales
    with area, so each tile lands on the official workload's operating
    point). Returns [C,F,H,W], the pipeline's own output rank.
    """
    batched = lq_video.dim() == 5
    lq = lq_video[0] if batched else lq_video
    C, F, H, W = lq.shape
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_kwargs.setdefault("num_frames", F)  # the canvas's own frame count
    tiles = plan_tiles(W, H, target=tile_target, threshold=tile_threshold,
                       overlap=overlap)
    if len(tiles) == 1:
        return pipe(LQ_video=lq_video.to(dev), height=H, width=W, **pipe_kwargs)

    num_frames = pipe_kwargs.pop("num_frames")
    base_topk = pipe_kwargs.pop("topk_ratio", None)
    out = None
    for x, y, w, h in tiles:
        crop = lq[:, :, y:y + h, x:x + w].to(dev, non_blocking=True)
        frames = pipe(
            LQ_video=crop.unsqueeze(0) if batched else crop,
            num_frames=num_frames, height=h, width=w,
            **({"topk_ratio": base_topk * (W * H) / (w * h)}
               if base_topk is not None else {}),
            **pipe_kwargs,
        )  # [C, F', h, w] in [-1, 1]
        if out is None:
            out = torch.zeros((C, frames.shape[-3], H, W), dtype=torch.float32)
        elif out.shape[1] != frames.shape[-3]:
            raise RuntimeError(f"tile frame counts diverge: {out.shape[1]} vs "
                               f"{frames.shape[-3]}")
        out[:, :, y:y + h, x:x + w] += frames.float().cpu() * tile_weight(x, y, tiles)
        del frames
    return out.to(dtype=lq_video.dtype)
