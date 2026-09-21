# Kernel reference

API and numerics contracts. All signatures below are the real ones in `flashvsr_sm89_ops/`.

Everything assumes bf16 activations on CUDA, contiguous tensors, and inference (all converted params are `requires_grad=False`).

---

## `fused_rms_rope` — RMSNorm + RoPE fusion (bf16 Triton)

```python
from flashvsr_sm89_ops import fused_rms_rope
fused_rms_rope(x, weight, freqs_cis, eps=1e-6, rope=True, head_dim=128)
# x: [B, L, D] bf16 → RMSNorm(weight) → [×RoPE]; freqs_cis = complex tensor with tail dim D//2
```

One Triton program per row: load row → fp32 RMS reduction → apply weight → interleave RoPE rotation with the `offs^1` partner-lane trick (pairs live in the same 128B L1 line, so the swap never leaves L1).

- **Numerics**: fp32 accumulation; norm output is 1–2 bf16 ulp from eager (max diff 0.016–0.031 on tested L, fraction > 0.01 ≈ 5e-7). Not bitwise — eager does an extra bf16 round-trip between norm and rope that the fusion removes.
- **Measured**: 6.65× (L=8448) to 14× (L≥18432) vs the eager chain; 848 GB/s at L=18432. ≈ −450 ms per full FlashVSR video (30 layers × 2 norm sites).
- `freqs_cis` split (`_freqs_cos_sin`) costs ~0.1 ms per call — cache `cos/sin` at the call site if you invoke per-layer.

## `fused_ln_modulate` / `fused_gate_add` — AdaLN fusion (bf16 Triton)

```python
from flashvsr_sm89_ops import fused_ln_modulate, fused_gate_add
fused_ln_modulate(x, scale, shift, eps=1e-6)   # = LayerNorm(x) * (1+scale) + shift
fused_gate_add(x, gate, residual)              # = x + gate * residual
```

- ⚠️ **Argument order: `scale` BEFORE `shift`.** The upstream `modulate()` convention is `(shift, scale)`. A reversed call order surfaces as a block A/B divergence that looks like a numerics bug — check wiring first.
- **Numerics**: the kernel replicates eager's per-op bf16 rounding chain (`bf16(LN) → ×(1+s) → bf16`), which makes `fused_gate_add` **bitwise identical** (max diff 0.0 at L=8448 and 18432) and `fused_ln_modulate` 1-ulp noise. A single fp32 rounding fails parity.
- **Measured**: LN 2.23–3.12×, gate 2.0–1.68× per instance; up to 2551 GB/s. ≈ −60 ms per full video across 30 blocks × 4 sites.
- `fused_gate_add` uses a flat 1D grid with param indexing `p = (idx // LD) * D + idx % D` (BLOCK=4096); `fused_ln_modulate` uses one program per row with `BLOCK = next_power_of_2(D)` — the reduction dim is a `constexpr`, never autotuned.

## `quantize_fp8` / `FP8Linear` — FP8 E4M3 linears

```python
from flashvsr_sm89_ops import quantize_fp8, FP8Linear, convert_linears_fp8
x8, scale = quantize_fp8(x2d)                    # fused absmax (atomic_max, fp32) + cast, 1 launch
lin = FP8Linear(nn.Linear(K, N))                 # wraps a bf16 Linear: static weight scale, E4M3 weights
n = convert_linears_fp8(model, parts=("ffn", "self", "cross"))
```

- Backend is `torch._scaled_mm` (cuBLASLt) with per-tensor scales — the only mode torch 2.6 supports on sm_89. An autotuned Triton FP8 GEMM measured 188–205 TF vs cuBLASLt's 204–300 TF on this pipeline's shapes; the data is in `benchmarks/fp8_gemm_bench.json`.
- **Scales**: weights static (calibrated once at conversion), activations dynamic (absmax per call). Measured GEMM speedup 2.07× at M=18432 (276–292 TF), 1.35–1.41× at M=55296 (~204 TF). Attention and norms stay bf16.
- `convert_linears_fp8` matches submodules by name (`ffn`/`self`/`cross`); it returns the number of converted linears.
- **Gotcha**: if the host model wraps modules (e.g. `enable_vram_management` wraps norms as `AutoWrappedModule`), unwrap via `getattr(m, "module", m)` before touching `.weight`/`.eps`. `FP8Linear` *subclasses* `nn.Linear`, so it survives such wrappers — the norm/eps paths are the ones that break.

## `FP8FFN` — GELU→FP8 fused mid-section

```python
from flashvsr_sm89_ops import FP8FFN, convert_ffn_fp8
n = convert_ffn_fp8(model)   # requires linears already FP8-converted ("ffn" part)
```

Replaces the FFN `Sequential` with a module that runs: quantize input → `_scaled_mm` (up) → `fused_gelu_quant_fp8` (GELU-tanh + absmax + E4M3 cast in **two** Triton launches) → `_scaled_mm` (down).

- **Numerics**: GELU uses the exp identity `1 − 2/(exp(2u)+1)` (no `tl.math.tanh` in Triton 3.2), then explicitly round-trips through bf16 to match eager's intermediate rounding — output codes are **bitwise identical** to the two-pass chain (0 differing codes at M=18432 and 55296).
- **Measured**: FFN chain 1.16×/1.12× vs the previous FP8 chain, ≈ −235 ms per full video.
- **Gotcha**: module-level Python floats are *invisible* inside `@triton.jit` in Triton 3.2 — constants like `C0 = 0.044715` and `FP8_MAX` must be passed as `tl.constexpr` arguments.

## `lcsa/` — Triton block-sparse attention (online softmax)

Validated alternative to the official CUDA BSA kernel: 104–116 TF vs 101–119 TF (±3%, same keep masks), max diff ≤ 1e-3, allclose on all tested (Nq, Nk, density) points.

- **Layout contract**: input is in *window order* `[1, N_win*128, H*D]`, i.e. after `WindowPartition3D`. The 128 tokens of a window are strided across raster order, not contiguous. Most mismatches trace to a window-vs-raster assumption, not the math; `debug_lcsa.py` does three-way bisection (kernel / reference / torch-CSR).
- Correctness bar: max_abs ≤ 1e-2 vs `lcsa/reference.py` in bf16.
- CSR mask form also avoids materializing the `[h, Lq, Lk]` boolean mask (~VRAM savings at 24 GB-class cards).

## TCDecoder (integration-level, not packaged as a kernel)

`channels_last` on the decoder: −16.7% decode wall time, bitwise-identical output, −0.5 GB peak. Optional `torch.compile(..., mode="max-autotune-no-cudagraphs")` on the MemBlocks: +1.06×, +0.95 GiB, LPIPS-identical. **CUDA Graphs are unsafe here** (the stream cache holds outputs across iterations) — use `no-cudagraphs` modes only.
