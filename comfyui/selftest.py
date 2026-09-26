#!/usr/bin/env python3
"""Drive the node end-to-end outside a running ComfyUI (GPU + weights needed).

Uses a ComfyUI checkout (COMFYUI_ROOT) for the comfy_api package, stubs
folder_paths, builds a VIDEO from raw frames (exactly what the core Load
Video node produces), and runs FlashVSRUpscale.execute through its public
surface — video + target_resolution, nothing else — checking output shape,
frame count, and the saved file.

    COMFYUI_ROOT=/root/ComfyUI python comfyui/selftest.py /path/smoke_input.mp4
"""
import os
import sys
import types
from fractions import Fraction

sys.path.insert(0, os.environ.get("COMFYUI_ROOT", "/root/ComfyUI"))
# the repository root carries both the comfyui package and flashvsr_sm89_ops
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio
import numpy as np
import torch

sys.modules["folder_paths"] = types.SimpleNamespace(
    # mirror the core semantic: <ComfyUI>/models — so a run inside a real
    # deployment discovers its checkout/weights instead of re-fetching them
    models_dir=os.environ.get("SELFTEST_MODELS_DIR",
                              os.path.join(os.environ.get("COMFYUI_ROOT", "/root/ComfyUI"),
                                           "models")),
    get_output_directory=lambda: os.environ.get("SELFTEST_OUT", "/tmp/comfy_selftest_out"),
)

import comfyui.nodes as node_mod  # noqa: E402  (this repo's node pack)
from comfy_api.latest import InputImpl, Types  # noqa: E402


def _make_video(images, fps):
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, frame_rate=fps))


def _expected_hw(W, H, target_h, multiple=128):
    """Mirror _image_batch_to_lq: pre-scale so 4x lands near target_h, snap."""
    scale = target_h / H
    w0 = max(1, round(W * scale / 4.0))
    h0 = max(1, round(H * scale / 4.0))
    return (round(h0 * 4) // multiple * multiple,
            round(w0 * 4) // multiple * multiple)


def _check(out_video, images, fps, tag, target_resolution):
    comps = out_video.get_components()
    f_in, H, W = images.shape[0], images.shape[1], images.shape[2]
    tH, tW = _expected_hw(W, H, node_mod.FlashVSRUpscale._TARGET_HEIGHTS[target_resolution])
    assert comps.images.shape[1] == tH, \
        f"{tag}: height {comps.images.shape[1]} != snapped bucket {tH}"
    assert comps.images.shape[2] == tW, \
        f"{tag}: width {comps.images.shape[2]} != snapped bucket {tW}"
    assert comps.images.shape[1] % 128 == 0 and comps.images.shape[2] % 128 == 0, \
        f"{tag}: output must sit on the 128px grid"
    assert comps.images.min() >= 0.0 and comps.images.max() <= 1.0, \
        f"{tag}: IMAGE contract violated"
    assert Fraction(comps.frame_rate) == fps, f"{tag}: frame rate must carry through"
    path = f"/tmp/selftest_flashvsr_{tag}.mp4"
    out_video.save_to(path, format=Types.VideoContainer("mp4"),
                      codec=Types.VideoCodec("h264"))
    r = imageio.get_reader(path)
    n, size = r.count_frames(), r.get_meta_data().get("size")
    r.close()
    print(f"[selftest] {tag}: {f_in}f {W}x{H} -> {target_resolution} = {n}f {size} "
          f"@ {float(fps):g} fps -> {path}")
    return n


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "input/example0_input.mp4"

    schema = node_mod.FlashVSRUpscale.GET_SCHEMA()
    assert schema.node_id == "FlashVSRUpscale", schema.node_id
    inputs = {i.id for i in schema.inputs}
    assert inputs == {"video", "target_resolution"}, inputs
    print(f"[selftest] schema ok: {schema.node_id} — inputs {sorted(inputs)}")

    r = imageio.get_reader(src)
    frames = np.stack([f for f in r]).astype(np.float32) / 255.0
    fps = Fraction(round(r.get_meta_data().get("fps", 30)))
    r.close()
    images = torch.from_numpy(frames)  # [F,H,W,C] 0..1
    print(f"[selftest] input {tuple(images.shape)} @ {float(fps):g} fps")

    up = node_mod.FlashVSRUpscale
    out = up.execute(video=_make_video(images, fps), target_resolution="1080p")
    n = _check(out.args[0], images, fps, "1080p", "1080p")

    # a second bucket proves the pre-scale path; the short clip also exercises
    # the last-frame padding (model needs F >= 25)
    out1 = up.execute(video=_make_video(images[:16], fps), target_resolution="720p")
    _check(out1.args[0], images[:16], fps, "720p", "720p")

    # between runs the pipeline must park in CPU RAM and still reproduce the
    # exact same output after being brought back
    assert torch.cuda.memory_allocated() < 3 * 2**30, \
        f"pipeline not parked: {torch.cuda.memory_allocated() / 2**30:.1f} GiB resident"
    out2 = up.execute(video=_make_video(images, fps), target_resolution="1080p")
    d = (out.args[0].get_components().images
         - out2.args[0].get_components().images).abs().max().item()
    assert d < 1e-3, f"park/warm roundtrip changed the output: max_abs_diff={d}"
    print(f"[selftest] park/warm roundtrip exact (max_abs_diff={d})")
    print(f"[selftest] PASS ({n}-frame 1080p output verified via imageio)")


if __name__ == "__main__":
    main()
