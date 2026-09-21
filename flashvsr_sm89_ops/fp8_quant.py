"""Fused FP8 (E4M3) dynamic quantization (Triton, sm_89).

`quantize_fp8(x2)` computes `s = amax(|x|)/448; (x2/s).to(e4m3)` in two
launches (absmax reduction with fp32 atomics, then quantize), keeping the
scale on the GPU — no host sync.

Input must be contiguous, shaped [M, K] (reshape before calling).
"""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _absmax_kernel(x_ptr, amax_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x), axis=0)
    tl.atomic_max(amax_ptr, m)


@triton.jit
def _quant_kernel(x_ptr, amax_ptr, out_ptr, n, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    amax = tl.load(amax_ptr)
    # divide by s; multiplying by 1/s breaks bitwise parity with eager
    s = tl.maximum(amax, 1e-12) / FP8_MAX
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.minimum(tl.maximum(x / s, -FP8_MAX), FP8_MAX)  # saturate: e4m3fn overflow -> NaN
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def _num_warps(block: int) -> int:
    return 4 if block <= 2048 else (8 if block <= 8192 else 16)


def quantize_fp8(x2: torch.Tensor):
    """x2: [M, K] contiguous bf16 -> (fp8 tensor, fp32 scale as GPU scalar)."""
    n = x2.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x2.device)
    if n == 0:
        return x2.to(torch.float8_e4m3fn), amax
    BLOCK = 8192 if n >= 8192 else triton.next_power_of_2(n)
    grid = (triton.cdiv(n, BLOCK),)
    _absmax_kernel[grid](x2, amax, n, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    x8 = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
    _quant_kernel[grid](x2, amax, x8, n, FP8_MAX=FP8_MAX, BLOCK=BLOCK,
                        num_warps=_num_warps(BLOCK))
    s = amax / FP8_MAX
    return x8, s
