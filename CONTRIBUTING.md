# Contributing

A kernel change is accepted when it survives the three gates below. A negative result with clean evidence is also a valid contribution.

## The three gates (every change)

1. **Op-level parity** vs the eager reference, on the real operating shapes *and* irregular ones (masking bugs hide at non-power-of-2 sizes).
   - bitwise where reachable (gate/add, quantize codes, channels_last);
   - otherwise ≤ 1–2 bf16 ulp (fused norms), max_abs ≤ 1e-2 for attention vs `reference.py`.
2. **Block-level A/B** through a real `DiTBlock` — max diff ≤ 1–2 ulp on the block output. This gate is non-negotiable: op benches can't catch argument-order/wiring bugs because your bench shares the wrapper's convention.
3. **End-to-end**: latency table on the pinned workload + quality gate (LPIPS ≤ 0.05, PSNR reported). Both must hold; "faster but blurry" is a rejection.

Record raw numbers as JSON next to the existing files in `benchmarks/` — same schema, no screenshots of numbers.

## Environment

- Python ≥ 3.10, torch ≥ 2.6 (CUDA 12.x), Triton ≥ 3.2, an sm_89 GPU for anything FP8 (bf16 kernels run on sm_80+).
- `pip install -e .` then `python -c "import flashvsr_sm89_ops"` as the smoke test.

## Implementation rules

- Accumulate reductions in fp32; round to bf16 **where eager rounds** — the bitwise parities in `docs/kernels.md` depend on the explicit casts.
- Never autotune a `constexpr` that controls a reduction dimension (`BLOCK_D`, etc.) — silent wrong results.
- Module-level Python float constants are invisible inside `@triton.jit` (Triton 3.2); pass them as `tl.constexpr` args.
- `tl.math.tanh` doesn't exist in Triton 3.2 — use `1 - 2/(exp(2u)+1)`.
- Performance claims need the kernel-truth: a chrome-trace kernel-name aggregation (not `key_averages()`, which double-counts), or a microbench at the real operating shapes.

## PR expectations

- One operator (or one rejected optimization) per PR.
- Include: parity numbers, A/B diff, e2e delta, quality-gate output, hardware used.
- Update `docs/kernels.md` if the numerics contract or API changes.
