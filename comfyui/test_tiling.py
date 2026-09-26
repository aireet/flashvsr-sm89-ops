#!/usr/bin/env python3
"""Offline tests for flashvsr_sm89_ops.tiling — CPU only, no weights needed.

    python comfyui/test_tiling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flashvsr_sm89_ops.tiling import (MULT, OVERLAP, TILE_TARGET, TILE_THRESHOLD,
                                      plan_tiles, render_tiled, tile_weight)


def check_plan(W, H, tiles, tiled_expected):
    assert (len(tiles) > 1) == tiled_expected, (W, H, tiles)
    for x, y, w, h in tiles:
        assert w % MULT == 0 and h % MULT == 0, (W, H, (x, y, w, h))
        if tiled_expected:
            assert w * h <= TILE_TARGET, (W, H, (x, y, w, h))
        assert 0 <= x and x + w <= W and 0 <= y and y + h <= H, (W, H, (x, y, w, h))
    # every canvas pixel covered by at least one tile
    cov = torch.zeros(H, W)
    for x, y, w, h in tiles:
        cov[y:y + h, x:x + w] += 1
    assert (cov >= 1).all(), f"{W}x{H}: uncovered pixels"
    # neighbouring tiles on the split axis share >= OVERLAP px
    xs = sorted({t[0] for t in tiles})
    for a, b in zip(xs, xs[1:]):
        assert a + tiles[0][2] - b >= OVERLAP, (W, H, xs)


def test_buckets():
    check_plan(1280, 640, plan_tiles(1280, 640), False)    # 720p: full frame
    check_plan(1920, 1024, plan_tiles(1920, 1024), False)  # 1080p: full frame
    check_plan(2560, 1408, plan_tiles(2560, 1408), True)   # 2K
    t4k = plan_tiles(3840, 2048)                           # 4K
    check_plan(3840, 2048, t4k, True)
    print(f"[tiling] 2K -> {plan_tiles(2560, 1408)}")
    print(f"[tiling] 4K -> {t4k}")


def test_sweep():
    for W in range(640, 4097, 128):
        for H in range(512, 2177, 128):
            tiles = plan_tiles(W, H)
            check_plan(W, H, tiles, W * H > TILE_THRESHOLD)


def test_weights_partition_of_unity():
    for W, H in [(2560, 1408), (3840, 2048), (1728, 2304), (4096, 2304)]:
        tiles = plan_tiles(W, H)
        if len(tiles) == 1:
            continue
        acc = torch.zeros(H, W)
        for x, y, w, h in tiles:
            acc[y:y + h, x:x + w] += tile_weight(x, y, tiles)
        assert torch.allclose(acc, torch.ones(H, W), atol=1e-5), \
            f"{W}x{H}: weight sum in [{acc.min():.4f}, {acc.max():.4f}]"
    print("[tiling] weights sum to 1 over every tiled canvas tested")


def test_render_identity():
    """A pipeline that returns its LQ crop unchanged must come back exact."""
    calls = []

    def fake_pipe(LQ_video, num_frames, height, width, **kw):
        lq = LQ_video[0] if LQ_video.dim() == 5 else LQ_video
        C, F, h, w = lq.shape
        calls.append((h, w, num_frames, kw.get("topk_ratio")))
        return lq  # like the real pipeline: [C,F,H,W] out

    torch.manual_seed(0)
    canvas = torch.rand(3, 25, 1408, 2560) * 2 - 1  # 2K, [-1,1]
    out = render_tiled(fake_pipe, canvas, num_frames=25, prompt="",
                       topk_ratio=2.0 * 768 * 1280 / (2560 * 1408))
    assert out.shape == canvas.shape
    d = (out.float() - canvas).abs().max().item()
    assert d < 1e-4, f"identity render drifted by {d}"
    assert len(calls) == 2 and all(c[1] < 2560 and c[0] == 1408 for c in calls), calls
    assert all(c[2] == 25 for c in calls), calls
    # per-tile topk keeps the canvas's token budget (area-scaled)
    assert abs(calls[0][3] - calls[1][3]) < 1e-9, calls
    # a [1,C,F,H,W] canvas comes back as the pipeline's own rank [C,F,H,W]
    calls.clear()
    out5 = render_tiled(fake_pipe, canvas.unsqueeze(0), num_frames=25, prompt="")
    assert out5.dim() == 4 and out5.shape == canvas.shape, out5.shape
    print(f"[tiling] identity render exact (max diff {d:.2e}), calls {calls}")


if __name__ == "__main__":
    test_buckets()
    test_sweep()
    test_weights_partition_of_unity()
    test_render_identity()
    print("[tiling] PASS")
