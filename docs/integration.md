# Integrating into a FlashVSR v1.1 pipeline

All wiring happens once, after the pipeline and VRAM management are set up, and is env-flag controlled so you can A/B any operator. Reference implementation: the `init_pipeline()` patch in the upstream workspace (`examples/WanVSR/infer_flashvsr_v1.1_tiny.py`).

```python
import os, torch
from flashvsr_sm89_ops import (
    fused_rms_rope, fused_ln_modulate, fused_gate_add,
    convert_linears_fp8, convert_ffn_fp8,
)
import diffsynth.models.wan_video_dit as dit   # upstream DiT module

def apply_flashvsr_sm89_ops(pipe):
    # 1) TCDecoder: channels_last (bitwise-identical, -16.7% decode)
    if os.environ.get("FS89_TCDEC_CL", "1") == "1":
        pipe.TCDecoder.to(memory_format=torch.channels_last)

    # 2) optional: compile MemBlocks (+1.06x, +0.95 GiB; do NOT enable cudagraphs here)
    if os.environ.get("FS89_TCDEC_COMPILE", "0") == "1":
        from utils.TCDecoder import MemBlock
        for m in pipe.TCDecoder.modules():
            if isinstance(m, MemBlock):
                m.forward = torch.compile(m.forward,
                                          mode="max-autotune-no-cudagraphs", dynamic=False)

    # 3) FP8 linears: ffn / self / cross (comma-separated; "" disables)
    parts = [p for p in os.environ.get("FS89_FP8", "ffn,self,cross").split(",") if p]
    if parts:
        convert_linears_fp8(pipe.denoising_model(), parts=parts)
        if "ffn" in parts and os.environ.get("FS89_FP8_FFNFUSED", "1") == "1":
            convert_ffn_fp8(pipe.denoising_model())   # needs "ffn" converted first

    # 4) inject fused bf16 ops into the DiT (upstream reads these hooks; add the
    #    same two branches shown in the reference patch if your checkout lacks them)
    if os.environ.get("FS89_FUSED_ROPE", "1") == "1":
        dit.FUSED_ROPE_FN = fused_rms_rope
    if os.environ.get("FS89_FUSED_ADALN", "1") == "1":
        dit.FUSED_ADALN = (fused_ln_modulate, fused_gate_add)

apply_flashvsr_sm89_ops(pipe)
pipe.init_cross_kv(); pipe.load_models_to_device(["dit", "vae"])
```

## Order matters

1. `channels_last` **after** `enable_vram_management` / any `.to(device, dtype)` — dtype/device migration resets layout.
2. `convert_linears_fp8` **before** `convert_ffn_fp8` (the fused FFN asserts its two linears are already `FP8Linear`).
3. Hook injection (`FUSED_ROPE_FN` / `FUSED_ADALN`) any time before the first forward; the upstream DiT branches on them per block.

## Env flags

| Flag | Default | Effect |
|---|---|---|
| `FS89_TCDEC_CL` | `1` | TCDecoder channels_last |
| `FS89_TCDEC_COMPILE` | `0` | MemBlock `torch.compile` (max-autotune-no-cudagraphs) |
| `FS89_FP8` | `ffn,self,cross` | FP8 parts; empty string = off |
| `FS89_FP8_FFNFUSED` | `1` | fused GELU→FP8 FFN (requires `ffn` in `FS89_FP8`) |
| `FS89_FUSED_ROPE` | `1` | RMSNorm+RoPE fusion |
| `FS89_FUSED_ADALN` | `1` | LN+modulate / gate+add fusion |

## If your DiT doesn't have the hooks

The upstream `DiTBlock.forward` needs ~10 lines to branch on `FUSED_ADALN` / `FUSED_ROPE_FN` (see the reference patch). Semantics to get right:

- `modulate(x, shift, scale) = x*(1+scale) + shift` — and our wrapper takes **(scale, shift)**, the reverse of the upstream call-site order.
- `gate`: `x = x + gate * attn_out` (gate applied to the residual branch, added to `x`).
- If the model is wrapped by VRAM management, unwrap norms with `getattr(m, "module", m)` before reading `.weight`/`.eps`.

Verify with a one-block A/B (max diff should be ≤ 1-2 bf16 ulp) before a full run — an op-level bench alone will NOT catch argument-order wiring bugs, because your bench and your wrapper share the same (possibly wrong) convention.

## Output range contract (saving frames yourself)

The pipeline returns video tensors in **[-1, 1]**. The official `tensor2video` maps them with `(x + 1) * 127.5` — use it, or this equivalent:

```python
if out.min() < -0.05:            # [-1,1] — the pipeline contract
    out = (out + 1.0) * 127.5
elif out.max() > 1.5:            # already [0,255]
    pass
else:                            # [0,1]
    out = out * 255.0
frames = out.clamp(0, 255).round().numpy().astype("uint8")
```

Do **not** guess the range with a bare `if out.max() > 1.5: out /= 255` heuristic: `[-1,1]` slips past it, and `clamp(0,1)` then crushes the entire dark half of every frame to black. This exact bug shipped in our demo exporter once — the quality gate still "passed" because reference and candidate were crushed identically. If you have an A/B-style check, validate its shared preprocessing once against a known-good external reference.

Note: the operators in this pack are numerics-preserving (bitwise or ≤2 bf16 ulp vs eager), so they never change this contract — the range is set by the FlashVSR pipeline itself.
