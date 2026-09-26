# Benchmarks: measurements and reproduction

Every number in the README traces to a JSON file in [`benchmarks/`](../benchmarks/). This page gives the commands. Hardware used: RTX 4090 D (sm_89), torch 2.6.0+cu124, Triton 3.2.0.

## Discipline

- Timing: CUDA events, warm-up runs before measurement, medians reported; end-to-end runs use full videos.
- Every claim has three levels of evidence: op-level parity JSON → block-level A/B → end-to-end time + quality gate.
- Negative results are kept (`*_triton_bench`) — they document why some paths were rejected.

## End-to-end A/B (the headline table)

Pin the workload first: pre-scale inputs to `352×192` so ×4 upscale lands exactly on the paper's 1408×768. Otherwise the official script picks target resolution from the source and you benchmark something else.

```bash
# baseline (all flags off)
FLASHVSR_TCDEC_CL=0 FLASHVSR_FP8= FLASHVSR_FUSED_ROPE=0 FLASHVSR_FUSED_ADALN=0 \
FLASHVSR_TCDEC_COMPILE=0 PINNED_LQ=352x192 python bench_e2e.py --out headtohead_orig.json

# optimized (defaults)
PINNED_LQ=352x192 python bench_e2e.py --out headtohead_opt2.json
```

Reference results (same session, RTX 4090 D): `headtohead_orig.json` 7749/8468/8487/7066 ms, 11.4–11.5 FPS, 13.1–13.9 GB peak → `headtohead_opt2.json` 5952/6505/6537/5443 ms, 14.8–15.0 FPS, 11.3–12.0 GB. **1.30×.**

1080p-class (input 480×270 → 1920×1024): optimized 14.26 s → 10.58 s per clip, 18.8 GB peak.

## Op-level suites (ship in this repo's heritage; port paths as needed)

| Suite | Command | Result file | Headline |
|---|---|---|---|
| RoPE fusion | `python fused_rope_bench.py` | `fused_rope_bench.json` | 6.6–14× per instance |
| AdaLN fusion | `python fused_adaln_bench.py` then `fused_adaln_block_smoke.py` | `fused_adaln_bench.json` | 2.0–3.1×, gate bitwise |
| FP8 GEMM (cuBLASLt) | `python fp8_gemm_bench.py` | `fp8_gemm_bench.json` | 2.07× @M=18k |
| Triton FP8 GEMM (rejected) | `python fp8_gemm_triton_bench.py` | `fp8_gemm_triton_bench.json` | 188–205 TF < cuBLASLt |
| FFN GELU→FP8 | `python fp8_ffn_bench.py` | `fp8_ffn_bench.json` | codes bitwise-identical |
| LCSA vs official BSA | `python bsa_compare.py` | `bsa_compare.json` | ±3% TF, allclose |
| TCDecoder layouts | `python tcdec_bench.py` | `tcdec_bench.json` | channels_last −16.7%, bitwise |
| TCDecoder compile | `python tcdec_compile_bench.py` | `tcdec_compile_bench.json` | 1.06×, +0.95 GiB |

> The op-level scripts live in the upstream workspace's `eval/` directory rather than this package — the JSONs here are the frozen evidence. Copying the harnesses in is a welcome first contribution (see CONTRIBUTING.md).

## Quality gate (run after ANY operator change)

Generate outputs for ≥ 2 clips with the patched pipeline, compare against official-pipeline outputs:

```bash
python gen_cand.py <outdir> example0 example3   # candidate outputs
python quality_ref.py cmp <outdir> example0 example3
# pass: LPIPS ≤ 0.05 vs reference; PSNR reported alongside (we got 38.3/36.0 dB)
```

Current evidence: [`benchmarks/quality_cmp.json`](../benchmarks/quality_cmp.json) — LPIPS 0.0108 / 0.0122, PSNR 38.28 / 35.99 dB.

> ⚠️ When saving pipeline output yourself: the tensor is **[-1, 1]** (official `tensor2video` maps it with `(x+1)*127.5`). A `max > 1.5 → divide by 255` heuristic silently misreads it as [0,1] and clamps the entire dark half of every frame to black — the gate still "passes" because reference and candidate get crushed identically, but the videos come out very dark. Detect the range from `min < 0`, don't guess.

## Profiling notes (things that skew results)

- `torch.profiler` `key_averages()` **double-counts** overlapping kernels — aggregate the chrome trace by kernel name instead.
- Bucketing trace kernels by name substring mis-buckets: cuDNN conv fprop kernels carry `cutlass` in their names, and the flash-attention kernel carries `cutlass` too. We briefly "found" a mystery 0.8s bf16 GEMM this way; there wasn't one.
- `triton.testing.do_bench` for anything under ~1 ms; CUDA events for pipeline-level.

## Quickstart end-to-end (pristine-clone A/B)

[`examples/run_flashvsr.py`](../examples/run_flashvsr.py) runs the pack on a
pristine upstream clone with zero source edits; `--no-ops` runs the stock
pipeline in the same process for A/B:

```bash
git clone https://github.com/OpenImagingLab/FlashVSR /tmp/FlashVSR
python examples/run_flashvsr.py --flashvsr-root /tmp/FlashVSR \
    --input assets/demo/example0_input.mp4 --out-dir /tmp/out
python examples/run_flashvsr.py --flashvsr-root /tmp/FlashVSR \
    --input assets/demo/example0_input.mp4 --out-dir /tmp/out_stock --no-ops
```

Evidence: [`benchmarks/quickstart_smoke.json`](../benchmarks/quickstart_smoke.json) —
same-session A/B, gen-only timing (input prep is CPU bicubic and is reported
separately; single runs, no repeats): stock 7640 ms / 11.13 FPS / 13.1 GB vs
pack **6137 ms / 13.85 FPS / 11.3 GB** with the zero-compile Triton attention
(~6045 ms with the CUDA BSA extension — backend delta 1.5-3%). That is
**1.25× same-session** against published session medians of 1.30×. Quality on
the same clip: PSNR 36.75 dB, LPIPS 0.017 vs stock; reruns are bit-identical
for a fixed seed.

### Fresh-session harness rerun (enable() path, 4 official clips)

[`benchmarks/pr2_harness_*.json`](../benchmarks/) — the published
`PINNED_LQ=352x192` methodology (warmup 1, iters 3, CUDA events around
`pipe()` only) re-run on a pristine upstream clone against `enable()`, in a
fresh process per config:

| config | FPS (4 clips) | peak VRAM | ratio vs same-session stock |
|---|---|---|---|
| stock (`--no-ops`) | 11.45–11.49 | 12.96–13.26 GB | 1.00× |
| `enable()`, CUDA BSA | 14.60–14.67 | 11.16–11.45 GB | 1.275–1.278× |
| `enable()`, Triton stub | 14.42–14.45 | 11.16–11.45 GB | 1.258–1.260× |

Stock FPS matches the published session (11.43–11.49) exactly; the pack
configs run ~2% under the published medians (14.84–14.96), so the 1.30×
published ratio reproduces as 1.26–1.28× same-session — session noise, and
the Triton-vs-CUDA-BSA delta stays at ~1.4%.
