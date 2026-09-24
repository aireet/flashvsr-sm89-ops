# Integrating into a FlashVSR v1.1 pipeline

## The short version

```python
import flashvsr_sm89_ops

# ... build the pipe exactly as the official example does, then after
pipe.to('cuda'); pipe.enable_vram_management(num_persistent_param_in_dit=None)
flashvsr_sm89_ops.enable(pipe)                     # <- the whole integration
pipe.init_cross_kv(); pipe.load_models_to_device(["dit", "vae"])
```

`enable()` applies, in dependency order:

1. TCDecoder -> channels_last (bitwise-identical, ~ −17% decode time)
2. optional MemBlock `torch.compile` (off by default: +0.95 GiB, slow warmup)
3. DiT Linears -> FP8 E4M3 via cuBLASLt, then the fused GELU→FP8 FFN
4. fused RMSNorm+RoPE and fused AdaLN — implemented by rebinding the DiT
   module's `SelfAttention.forward` / `CrossAttention.forward` /
   `DiTBlock.forward` with fused-aware copies. The upstream stock path is
   kept verbatim as the fallback branch, so every flag can be turned off
   without touching FlashVSR source, and re-running `enable()` is safe.
5. the block-sparse attention backend: `import flashvsr_sm89_ops` installs a
   Triton stand-in for the mit-han-lab `block_sparse_attn` module when that
   package is absent, so the upstream `from block_sparse_attn import ...`
   works with zero compilation. With the real package installed, the real
   kernel wins automatically.

`enable()` returns a dict of what it turned on, and prints one summary line.
It also verifies the DiT module structure it is about to rebind — run it once
interactively before scripting around it.

A complete, runnable integration is [`examples/run_flashvsr.py`](../examples/run_flashvsr.py):
it mirrors the official example 1:1 (same input prep, same pipeline call,
same output naming), adds weights auto-download, and needs no source edits.

## Import order

`import flashvsr_sm89_ops` **before** `diffsynth` — the Triton stand-in for
`block_sparse_attn` must be in `sys.modules` before the upstream DiT module
is imported. In your own scripts that means the import belongs at the top of
the file; `examples/run_flashvsr.py` does it for you.

## Env flags (read once, at `enable()` time)

| Flag | Default | Effect |
|---|---|---|
| `FS89_TCDEC_CL` | `1` | TCDecoder channels_last |
| `FS89_TCDEC_COMPILE` | `0` | MemBlock `torch.compile` (max-autotune-no-cudagraphs; needs the checkout's `examples/WanVSR/utils` importable) |
| `FS89_FP8` | `ffn,self,cross` | FP8 parts; empty string = off |
| `FS89_FP8_FFNFUSED` | `1` | fused GELU→FP8 FFN (requires `ffn` in `FS89_FP8`) |
| `FS89_FUSED_ROPE` | `1` | RMSNorm+RoPE fusion |
| `FS89_FUSED_ADALN` | `1` | LN+modulate / gate+add fusion |
| `FS89_LCSA` | `auto` | `auto`: real BSA if installed, else Triton stub; `triton`/`bsa` force a side |

`enable(pipe, parts=..., compile_memblocks=...)` overrides the first three
programmatically if you prefer arguments over env vars.

## Semantics the fused paths rely on

If you audit the rebound forwards against upstream, these are the contracts:

- `modulate(x, shift, scale) = x*(1+scale) + shift` — our fused wrapper takes
  **(scale, shift)**, the reverse of the upstream call-site order.
- `gate`: `x = x + gate * attn_out` (gate applied to the residual branch,
  added to `x`).
- If the model is wrapped by VRAM management, unwrap norms with
  `getattr(m, "module", m)` before reading `.weight`/`.eps`.

Verify with a one-block A/B (max diff should be ≤ 1-2 bf16 ulp) before a full
run — an op-level bench alone will NOT catch argument-order wiring bugs,
because your bench and your wrapper share the same (possibly wrong) convention.

## If you keep the old manual wiring

The env-flag patch described in earlier revisions (adding the two branch
blocks to `wan_video_dit.py` yourself) still works and produces the same
forward semantics; `enable()` just does it without the edits. Don't do both
with different flags — the rebind wins, and the flags are read from the
module attributes it sets.

## Output range contract (saving frames yourself)

The pipeline returns video tensors in **[-1, 1]**. The official `tensor2video`
maps them with `(x + 1) * 127.5` — use it, or this equivalent:

```python
if out.min() < -0.05:            # [-1,1] — the pipeline contract
    out = (out + 1.0) * 127.5
elif out.max() > 1.5:            # already [0,255]
    pass
else:                            # [0,1]
    out = out * 255.0
frames = out.clamp(0, 255).round().numpy().astype("uint8")
```

Do **not** guess the range with a bare `if out.max() > 1.5: out /= 255`
heuristic: `[-1,1]` slips past it, and `clamp(0,1)` then crushes the entire
dark half of every frame to black. If you run an A/B-style quality check, both
sides share the saving path — a range bug there cancels out and the gate still
passes. Validate the saving path once against a known-good external reference.

The operators in this pack are numerics-preserving (bitwise or ≤2 bf16 ulp vs
eager), so they never change this contract — the range is set by the FlashVSR
pipeline itself.

## Upstream clip-length floor

The streaming loop runs `(num_frames - 1) // 8 - 2` iterations, so upstream
fails with `torch.cat(): expected a non-empty list of Tensors` below
`num_frames = 25`. `examples/run_flashvsr.py` and the ComfyUI node hold the
last frame until the minimum is reached; if you roll your own runner, do the
same.
