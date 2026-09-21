# flashvsr-sm89-ops

Drop-in operator pack that runs [FlashVSR v1.1 (Tiny)](https://github.com/OpenImagingLab/FlashVSR) at **~1.3× on a single RTX 4090**, with **lower peak VRAM** and a measured, gated quality budget.

It packages five kernels/operators developed for the sm_89 (Ada) generation — two fused bf16 Triton pairs (attention input path, AdaLN path), FP8 E4M3 linears via cuBLASLt, a fused GELU→FP8 quantizer, plus a Triton block-sparse attention kernel that matches the official CUDA BSA kernel within noise — and the benchmark harnesses that justify every number below.

## Results

Same machine, same session, A/B: official FlashVSR pipeline vs all operators enabled.
RTX 4090 D, 1408×768 (the paper's 768×1408 workload), 1-step DMD, full videos. Raw data: [`benchmarks/headtohead_orig.json`](benchmarks/headtohead_orig.json) / [`headtohead_opt2.json`](benchmarks/headtohead_opt2.json).

| Clip | Frames | Baseline | Optimized | Δ | FPS (base → opt) | Peak VRAM (base → opt) |
|---|---|---|---|---|---|---|
| example0 | 89 | 7749 ms | **5952 ms** | −23.2% | 11.49 → **14.96** | 13.1 → **11.3** GB |
| example1 | 97 | 8468 ms | **6505 ms** | −23.2% | 11.46 → **14.91** | 13.8 → **12.0** GB |
| example2 | 97 | 8487 ms | **6537 ms** | −23.0% | 11.43 → **14.84** | 13.9 → **12.0** GB |
| example3 | 81 | 7066 ms | **5443 ms** | −23.0% | 11.47 → **14.88** | 13.4 → **11.7** GB |

**1.30× end-to-end, −1.9 GB peak VRAM.** Quality gate vs official outputs: LPIPS 0.0117 / 0.0133 (threshold ≤ 0.05), PSNR 37.2 / 36.4 dB ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)). At 1920×1024 input (1080p-class) the pack holds −25.8% latency at 18.8 GB peak — comfortably inside a 24 GB card.

## Demos — the official examples, original vs optimized

All four official example clips, rendered end-to-end on the same RTX 4090: the stock pipeline vs this pack (all operators on). Inputs are the official example videos pre-scaled to 352×192 (→ 1408×768 output, the timed workload).

| Case | Input | Stock pipeline (4090) | With this pack (4090) |
|---|---|---|---|
| **example0** · 89F · 7749 → **5952 ms** | ![input0](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/example0_input.mp4) | ![stock0](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/baseline_example0.mp4) | ![opt0](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/optimized_example0.mp4) |
| **example1** · 97F · 8468 → **6505 ms** | ![input1](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/example1_input.mp4) | ![stock1](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/baseline_example1.mp4) | ![opt1](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/optimized_example1.mp4) |
| **example2** · 97F · 8487 → **6537 ms** | ![input2](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/example2_input.mp4) | ![stock2](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/baseline_example2.mp4) | ![opt2](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/optimized_example2.mp4) |
| **example3** · 81F · 7066 → **5443 ms** | ![input3](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/example3_input.mp4) | ![stock3](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/baseline_example3.mp4) | ![opt3](https://github.com/aireet/flashvsr-sm89-ops/releases/download/demo-v0/optimized_example3.mp4) |

Latency: stock → optimized (medians, −23%): **1.30× end-to-end, 11.5 → 15.0 FPS**. Direct links: [example0](assets/demo/baseline/example0.mp4) · [example1](assets/demo/baseline/example1.mp4) · [example2](assets/demo/baseline/example2.mp4) · [example3](assets/demo/baseline/example3.mp4) (stock), same paths under `assets/demo/optimized/` for the pack.

Visual quality between the two columns is gated: LPIPS ≤ 0.05 and PSNR 37+ dB against the stock outputs ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)). The rendered demo videos themselves: `eval/demo_times_baseline.json` / `demo_times_optimized.json` record the single-render wall times (6996/6499/6524/5430 ms optimized) and peak VRAM (13.3 → 11.2–11.5 GB).

## Requirements

- GPU: RTX 4090 / 4090 D (sm_89). The FP8 path requires sm_89 tensor cores; the bf16 Triton kernels and the LCSA kernel run on anything sm_80+.
- Python ≥ 3.10, PyTorch ≥ 2.6 (CUDA 12.4), Triton ≥ 3.2.
- A working FlashVSR v1.1 checkout (this pack patches into its pipeline; it does not replace it). FlashVSR and its model assets are Apache-2.0 / per their upstream terms.

## Quickstart

```bash
pip install .            # from this directory; or just keep it on PYTHONPATH
python -c "import flashvsr_sm89_ops; print('ok')"
```

Standalone use of any operator:

```python
import torch
from flashvsr_sm89_ops import fused_ln_modulate, fused_gate_add, quantize_fp8

x = torch.randn(2, 4096, 1536, device="cuda", dtype=torch.bfloat16)
scale = torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16)
shift = torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16)

y = fused_ln_modulate(x, scale, shift)   # LayerNorm(x)*(1+scale)+shift — note the argument order
z = fused_gate_add(x, scale, x)          # x + gate * residual, bitwise-identical to eager
x8, s = quantize_fp8(x)                  # fused absmax + E4M3 cast, (fp8, per-tensor scale)
```

Wiring into the FlashVSR pipeline (env-flag controlled, all default-on after gating) is a ~40-line patch to the inference entry point — see [`docs/integration.md`](docs/integration.md). Per-kernel API, numerics contracts, and the wiring gotchas we hit: [`docs/kernels.md`](docs/kernels.md). Reproducing every number: [`docs/benchmarks.md`](docs/benchmarks.md).

## What's inside

| Operator | Replaces | Measured (RTX 4090 D) | Data |
|---|---|---|---|
| `fused_rms_rope` | RMSNorm → RoPE (2 kernels + 4 elementwise passes) | **6.6–14× per instance** (848 GB/s), ≈ −450 ms per full video | [`benchmarks/fused_rope_bench.json`](benchmarks/fused_rope_bench.json) |
| `fused_ln_modulate` / `fused_gate_add` | LayerNorm+modulate / gate+add (5 eager kernels each) | **2.0–3.1× per instance**; gate output bitwise-identical; ≈ −60 ms | [`benchmarks/fused_adaln_bench.json`](benchmarks/fused_adaln_bench.json) |
| `FP8Linear` (E4M3, per-tensor) | bf16 `nn.Linear` in DiT FFN/QKV/out-proj | GEMM **2.07×** (M=18k) / **1.35–1.41×** (M=55k), 276–292 TF on tensor cores | [`benchmarks/fp8_gemm_bench.json`](benchmarks/fp8_gemm_bench.json) |
| `FP8FFN` (GELU→FP8 fusion) | FFN mid-section: GELU + separate quantize pass | bitwise-identical codes vs the two-pass chain; **1.12–1.16×** chain, ≈ −235 ms | [`benchmarks/fp8_ffn_bench.json`](benchmarks/fp8_ffn_bench.json) |
| `TCDecoder channels_last` (+ optional MemBlock `torch.compile`) | NCHW decoder convs | **−16.7%** decode, bitwise-identical; compile adds 1.06× (+0.95 GiB) | [`benchmarks/tcdec_bench.json`](benchmarks/tcdec_bench.json) |
| `lcsa/` Triton block-sparse attention | official CUDA BSA kernel (sm_80 compat build) | **parity: 104–116 vs 101–119 TF**, max diff ≤ 1e-3, allclose | [`benchmarks/bsa_compare.json`](benchmarks/bsa_compare.json) |

The LCSA kernel is shipped as a validated drop-in *alternative* (it removed our build dependency on the external CUDA package and documents the exact layout contract), not as the source of the speedup — the official kernel is already at ~78% of achievable throughput on this chip.

## Correctness discipline

Every operator in this pack passed a three-level gate before integration, and the harnesses ship with it:

1. **Op-level parity** vs the eager reference — bitwise where reachable (`gate_add`, GELU→FP8 codes, channels_last), else bounded (≤ 1–2 bf16 ulp for fused norms).
2. **Block-level A/B** through a real `DiTBlock` (this is what catches wiring bugs op-level benches can't — see `docs/kernels.md` § "argument order").
3. **End-to-end quality gate** — full videos vs official outputs, LPIPS ≤ 0.05 and PSNR reported ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)).

## Known limits (measured, not guessed)

- Numbers are for the pinned 1408×768 / 1-step pipeline above; other resolutions shift the GEMM M-dimension and the FP8 speedups (see the M=18k vs M=55k rows).
- A custom Triton FP8 GEMM was **falsified** (188–205 TF vs cuBLASLt's 204–300 TF) — `torch._scaled_mm` stays the backend.
- CUDA Graphs were **falsified** for this pipeline (launch gaps < 2% of wall).
- Conv rewrites (Winograd) were **falsified** — full-res decoder convs sit at the memory-bandwidth wall, where extra FLOPs buy nothing.
- Raw JSON in [`benchmarks/`](benchmarks/); the negative results are kept deliberately.

## Acknowledgments & license

Built on [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (Wan2.1-1.3B, LCSA, TCDecoder), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention). This pack is new code under **Apache-2.0** (see [LICENSE](LICENSE)); upstream licenses govern the pipeline and model weights you run it with.
