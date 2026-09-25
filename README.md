# flashvsr-sm89-ops

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/torch-2.6%2B-ee4c2c)]()
[![Triton](https://img.shields.io/badge/triton-3.2%2B-blue)]()
[![GPU](https://img.shields.io/badge/GPU-RTX_4090_(sm89)-green)]()
[![arXiv](https://img.shields.io/badge/arXiv-2510.12747-b31b1b)](https://arxiv.org/abs/2510.12747)

Drop-in operator pack that runs [FlashVSR v1.1 (Tiny)](https://github.com/OpenImagingLab/FlashVSR) at **~1.3× on a single RTX 4090**, with **lower peak VRAM** and a measured, gated quality budget. The official pipeline targets A100-class GPUs for its real-time claim; this pack brings streaming VSR to a consumer card at **14.4–15.0 FPS @ 1408×768** — **without compiling a single CUDA extension and without editing FlashVSR source** ([quickstart](#quickstart); [ComfyUI nodes](https://github.com/aireet/ComfyUI-FlashVSR-SM89)).

It packages five operators for the sm_89 (Ada) generation — fused RMSNorm+RoPE, fused AdaLN (LN+modulate / gate+add), FP8 E4M3 linears via cuBLASLt, a fused GELU→FP8 FFN mid-section, and channels_last/compile handling for the TCDecoder — plus a Triton block-sparse attention kernel matching the official CUDA BSA kernel within noise, and the benchmark data behind every number below.

## Results

Same machine, same session, A/B: official FlashVSR pipeline vs all operators enabled.
RTX 4090 D, 1408×768 (the paper's 768×1408 workload), 1-step DMD, full videos. Raw data: [`benchmarks/headtohead_orig.json`](benchmarks/headtohead_orig.json) / [`headtohead_opt2.json`](benchmarks/headtohead_opt2.json).

| Clip | Frames | Baseline | Optimized | Δ | FPS (base → opt) | Peak VRAM (base → opt) |
|---|---|---|---|---|---|---|
| example0 | 89 | 7749 ms | **5952 ms** | −23.2% | 11.49 → **14.96** | 13.1 → **11.3** GB |
| example1 | 97 | 8468 ms | **6505 ms** | −23.2% | 11.46 → **14.91** | 13.8 → **12.0** GB |
| example2 | 97 | 8487 ms | **6537 ms** | −23.0% | 11.43 → **14.84** | 13.9 → **12.0** GB |
| example3 | 81 | 7066 ms | **5443 ms** | −23.0% | 11.47 → **14.88** | 13.4 → **11.7** GB |

**1.30× end-to-end, −1.9 GB peak VRAM.** Quality gate vs official outputs: LPIPS 0.0108 / 0.0122 (threshold ≤ 0.05), PSNR 38.3 / 36.0 dB ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)). At 1920×1024 input (1080p-class) the pack holds −25.8% latency at 18.8 GB peak — comfortably inside a 24 GB card.

An independent re-run of the same methodology in fresh processes against the one-call integration (pristine upstream clone, no dev-venv crutches) reproduced it as **1.26–1.28× same-session** (stock 11.45–11.49 FPS — identical to the published session; Triton attention backend 1.4% behind the CUDA one): [`benchmarks/pr2_harness_*.json`](benchmarks/), methodology in [`docs/benchmarks.md`](docs/benchmarks.md).

## Demos — the official examples, original vs optimized

![example0: stock vs this pack](assets/demo/stock_vs_pack_example0.gif)

All four official example clips, rendered end-to-end on the same RTX 4090: the stock pipeline vs this pack (all operators on). Inputs are the official example videos pre-scaled to 352×192 (→ 1408×768 output, the timed workload).

Click any card to play the video (GitHub's built-in player).

| Case | Input | Stock pipeline (4090) ▶ | With this pack (4090) ▶ |
|---|---|---|---|
| **example0** · 89F · 7749 → **5952 ms** | <a href="assets/demo/example0_input.mp4"><img src="assets/demo/thumbs/example0_input.jpg" width="170" alt="input0"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/baseline/example0.mp4"><img src="assets/demo/thumbs/baseline_example0.jpg" width="270" alt="stock0"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/optimized/example0.mp4"><img src="assets/demo/thumbs/optimized_example0.jpg" width="270" alt="opt0"></a> |
| **example1** · 97F · 8468 → **6505 ms** | <a href="assets/demo/example1_input.mp4"><img src="assets/demo/thumbs/example1_input.jpg" width="170" alt="input1"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/baseline/example1.mp4"><img src="assets/demo/thumbs/baseline_example1.jpg" width="270" alt="stock1"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/optimized/example1.mp4"><img src="assets/demo/thumbs/optimized_example1.jpg" width="270" alt="opt1"></a> |
| **example2** · 97F · 8487 → **6537 ms** | <a href="assets/demo/example2_input.mp4"><img src="assets/demo/thumbs/example2_input.jpg" width="170" alt="input2"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/baseline/example2.mp4"><img src="assets/demo/thumbs/baseline_example2.jpg" width="270" alt="stock2"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/optimized/example2.mp4"><img src="assets/demo/thumbs/optimized_example2.jpg" width="270" alt="opt2"></a> |
| **example3** · 81F · 7066 → **5443 ms** | <a href="assets/demo/example3_input.mp4"><img src="assets/demo/thumbs/example3_input.jpg" width="170" alt="input3"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/baseline/example3.mp4"><img src="assets/demo/thumbs/baseline_example3.jpg" width="270" alt="stock3"></a> | <a href="https://github.com/aireet/flashvsr-sm89-ops/blob/main/assets/demo/optimized/example3.mp4"><img src="assets/demo/thumbs/optimized_example3.jpg" width="270" alt="opt3"></a> |

Latency: stock → optimized (medians, −23%): **1.30× end-to-end, 11.5 → 15.0 FPS**. Direct links: [example0](assets/demo/baseline/example0.mp4) · [example1](assets/demo/baseline/example1.mp4) · [example2](assets/demo/baseline/example2.mp4) · [example3](assets/demo/baseline/example3.mp4) (stock), same paths under `assets/demo/optimized/` for the pack.

Visual quality between the two columns is gated: LPIPS ≤ 0.05 and PSNR ~36–38 dB against the stock outputs ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)). The rendered demo videos themselves: `eval/demo_times_baseline.json` / `demo_times_optimized.json` record the single-render wall times (6991/6482/6519/5422 ms optimized) and peak VRAM (13.3 → 11.2–11.5 GB).

## Requirements

- GPU: RTX 4090 / 4090 D (sm_89). The FP8 path requires sm_89 tensor cores; the bf16 Triton kernels and the LCSA kernel run on anything sm_80+.
- Python ≥ 3.10, PyTorch ≥ 2.6, Triton ≥ 3.2. Newer stacks work too — the pipeline is verified end-to-end on torch 2.14 / triton 3.8 / transformers 5.17.
- A FlashVSR v1.1 checkout to import the pipeline from. **No CUDA compilation of any kind** — the mit-han-lab Block-Sparse-Attention extension is not needed; a Triton stand-in is installed automatically when it's absent. Other import-time landmines in upstream diffsynth (`modelscope`, transformers-v5 renames) are defused by the pack as well.

## Quickstart

Three commands, no source edits. The reference runner mirrors the official example 1:1 (same input prep, same pipeline call, same output naming), auto-downloads weights (~6.5 GB, once, from HuggingFace) and prints per-clip timing:

> **Where to install from:** these commands use the `feat/quickstart-zero-compile` branch, which carries the quickstart (PR #2). After that PR merges, the plain repo URL works identically.

```bash
git clone https://github.com/OpenImagingLab/FlashVSR
git clone -b feat/quickstart-zero-compile https://github.com/aireet/flashvsr-sm89-ops
pip install ./flashvsr-sm89-ops

python flashvsr-sm89-ops/examples/run_flashvsr.py \
    --flashvsr-root ./FlashVSR \
    --input /path/to/video.mp4 \
    --out-dir ./results
```

`--input` accepts a video file or a directory of frames, and can be repeated for batches. Compare against the stock pipeline by adding `--no-ops` (same process, same weights). If weights are already in place at `FlashVSR/examples/WanVSR/FlashVSR-v1.1/`, add `--no-download` to skip the fetch. No clip handy? `flashvsr-sm89-ops/assets/demo/example0_input.mp4` (352×192, the input of the timed workload above) works.

**Or two lines in your own script** — anywhere after `enable_vram_management`, before `init_cross_kv()`:

```python
import flashvsr_sm89_ops
flashvsr_sm89_ops.enable(pipe)     # FP8 linears + fused kernels + channels_last + Triton attention
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

Integration details and env flags: [`docs/integration.md`](docs/integration.md). Per-kernel API and numerics contracts: [`docs/kernels.md`](docs/kernels.md). Reproducing every number: [`docs/benchmarks.md`](docs/benchmarks.md).

**ComfyUI?** There is a node pack: [ComfyUI-FlashVSR-SM89](https://github.com/aireet/ComfyUI-FlashVSR-SM89) — verified end-to-end on a live ComfyUI (torch 2.14 / triton 3.8 / transformers 5.17): load a clip, wire it into *FlashVSR Upscale 4x (sm89)*, save.

## What's inside

| Operator | Replaces | Measured (RTX 4090 D) | Data |
|---|---|---|---|
| `fused_rms_rope` | RMSNorm → RoPE (2 kernels + 4 elementwise passes) | **6.6–14× per instance** (848 GB/s), ≈ −450 ms per full video | [`benchmarks/fused_rope_bench.json`](benchmarks/fused_rope_bench.json) |
| `fused_ln_modulate` / `fused_gate_add` | LayerNorm+modulate / gate+add (5 eager kernels each) | **2.0–3.1× per instance**; gate output bitwise-identical; ≈ −60 ms | [`benchmarks/fused_adaln_bench.json`](benchmarks/fused_adaln_bench.json) |
| `FP8Linear` (E4M3, per-tensor) | bf16 `nn.Linear` in DiT FFN/QKV/out-proj | GEMM **2.07×** (M=18k) / **1.35–1.41×** (M=55k), 276–292 TF on tensor cores | [`benchmarks/fp8_gemm_bench.json`](benchmarks/fp8_gemm_bench.json) |
| `FP8FFN` (GELU→FP8 fusion) | FFN mid-section: GELU + separate quantize pass | bitwise-identical codes vs the two-pass chain; **1.12–1.16×** chain, ≈ −235 ms | [`benchmarks/fp8_ffn_bench.json`](benchmarks/fp8_ffn_bench.json) |
| `TCDecoder channels_last` (+ optional MemBlock `torch.compile`) | NCHW decoder convs | **−16.7%** decode, bitwise-identical; compile adds 1.06× (+0.95 GiB) | [`benchmarks/tcdec_bench.json`](benchmarks/tcdec_bench.json) |
| `lcsa/` Triton block-sparse attention | official CUDA BSA kernel (sm_80 compat build) | **parity: 104–116 vs 101–119 TF**, max diff ≤ 1e-3, allclose; end-to-end within ~3% of the CUDA kernel | [`benchmarks/bsa_compare.json`](benchmarks/bsa_compare.json), [`benchmarks/quickstart_smoke.json`](benchmarks/quickstart_smoke.json) |

The LCSA kernel is a validated drop-in *alternative* to the official CUDA kernel (which already runs at ~78% of achievable throughput on this chip), not the source of the speedup.

## Correctness discipline

Every operator passed a three-level gate before integration:

1. **Op-level parity** vs the eager reference — bitwise where reachable (`gate_add`, GELU→FP8 codes, channels_last), else ≤ 1–2 bf16 ulp (fused norms).
2. **Block-level A/B** through a real `DiTBlock` — catches wiring bugs op-level benches cannot.
3. **End-to-end quality gate** — full videos vs official outputs, LPIPS ≤ 0.05 and PSNR reported ([`benchmarks/quality_cmp.json`](benchmarks/quality_cmp.json)).

## FAQ

- **Do I need to build Block-Sparse-Attention?** No. Importing this pack installs a Triton stand-in for the `block_sparse_attn` module when the real package is absent, so the FlashVSR DiT imports cleanly on a stock 4090. If you do have the CUDA extension installed it wins automatically (`FS89_LCSA=auto`, default); `FS89_LCSA=triton|bsa` forces either side. Measured end-to-end delta between the two: ~1.5–3% ([`benchmarks/quickstart_smoke.json`](benchmarks/quickstart_smoke.json)).
- **Why does the install pull in `ftfy`?** It's needed by the text-encoder path (upstream lists it in its own requirements but it's easy to miss — so this pack declares it). Similarly, diffsynth imports `modelscope` at import time but never calls it when loading the release weights locally; the pack stubs it and raises a clear error only if the downloader path is ever actually used. Details in [`docs/integration.md`](docs/integration.md#compatibility-layers-installed-at-pack-import).
- **Which GPUs?** FP8 paths need sm_89 tensor cores (RTX 4090 / 4090 D, L40, RTX 6000 Ada). The bf16 Triton kernels and LCSA run on anything sm_80+ (A100, 3090, …); on non-Ada cards convert with `parts=()` and keep the fused norms + channels_last.
- **Other resolutions / models?** Numbers here are pinned to the 1408×768 / 1-step workload. The operators are shape-generic (per-tensor scales, no baked shapes); expect the FP8 GEMM gain to shift with the M dimension (see the M=18k vs M=55k rows).
- **Very short clips?** The streaming loop needs `num_frames ≥ 25` (upstream constraint: `(F-1)//8 - 2` iterations). `examples/run_flashvsr.py` and the ComfyUI node hold the last frame until the minimum is reached.
- **Timing scope?** Input prep (CPU bicubic ×4 + padding, ~1.5 s per clip) is reported separately from `pipe()` in the runner output — only the `pipe()` window is the comparable number. See `timing_scope` in [`benchmarks/quickstart_smoke.json`](benchmarks/quickstart_smoke.json).
- **Training?** No — inference only; all converted params are frozen.
- **Why isn't LCSA the speedup?** The official CUDA BSA kernel already runs at ~78% of achievable throughput on this chip; the wins are in the elementwise/norm/linear paths.

## Known limits

- Numbers are for the pinned 1408×768 / 1-step pipeline above; other resolutions shift the GEMM M-dimension and the FP8 speedups (see the M=18k vs M=55k rows).
- A custom Triton FP8 GEMM measured 188–205 TF vs cuBLASLt's 204–300 TF — `torch._scaled_mm` stays the backend.
- CUDA Graphs: launch gaps are < 2% of wall for this pipeline; not worth the integration complexity.
- Conv rewrites (Winograd): full-res decoder convs sit at the memory-bandwidth wall; extra FLOPs buy nothing.
- Raw JSON in [`benchmarks/`](benchmarks/).

## Acknowledgments & license

Built on [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (Wan2.1-1.3B, LCSA, TCDecoder), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention). This pack is new code under **Apache-2.0** (see [LICENSE](LICENSE)); upstream licenses govern the pipeline and model weights you run it with.
