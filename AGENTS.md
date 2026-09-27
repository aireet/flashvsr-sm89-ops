# AGENTS.md

Instructions for AI coding agents working in this repository.

## Scope

- Always: read `docs/kernels.md` before touching a kernel; run the three gates in `CONTRIBUTING.md` before claiming success; keep raw results as JSON in `benchmarks/`.
- Ask first: adding a new dependency; changing a public API signature in `flashvsr_sm89_ops/`.
- Never: enable cudagraphs on the TCDecoder/stream cache (unsafe with cross-iteration output reuse); autotune reduction-dim constexprs; delete or "clean up" the explicit bf16 rounding chain in kernels — it replicates eager numerics and the bitwise parities depend on it.

## Hardware/software assumptions

- Reference GPU: RTX 4090 (sm_89, 128 SM, 100 KB smem/SM, 1008 GB/s HBM, no TMA/clusters). FP8 requires sm_89; bf16 Triton kernels target sm_80+.
- torch ≥ 2.6 (CUDA 12.4), Triton 3.2 (no `tl.math.tanh`; module-level Python floats invisible inside `@triton.jit`).

## Acceptance commands

```bash
pip install -e . && python -c "import flashvsr_sm89_ops"          # smoke
# quickstart integration (needs $FLASHVSR_ROOT + weights; see examples/)
python examples/run_flashvsr.py --flashvsr-root "$FLASHVSR_ROOT" --input <clip> --no-download
# LCSA parity + perf (files use flat imports — run from inside the dir)
cd flashvsr_sm89_ops/lcsa && python test_correctness.py --perf
```

End-to-end and quality-gate commands live in `docs/benchmarks.md`; the frozen evidence JSONs live in `benchmarks/`. If your change moves a headline number by more than noise (±2%), regenerate the corresponding JSON and say so in the PR.
