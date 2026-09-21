#!/usr/bin/env python3
"""M3: FP8 动态量化的 Triton 融合 kernel（sm_89, triton 3.2）。

动机：eager 路径 abs→amax→div→cast 有 4 次显存往返（temp 物化，~365MB 流量 @56MB 输入，
仅跑 ~430GB/s），吃掉了 _scaled_mm 的全部收益。融合后理论流量 141MB（amax 读 1 次 +
量化读 1 写 0.5 次）→ 量化开销 ~0.34ms → ~0.15ms（M=18432,K=1536 工况）。

约定：x 必须 reshape(-1, K) 后 contiguous（reshape 不连续时会复制，安全）。
amax 用 fp32 原子归约单 pass 完成；scale 保持在 GPU 上，无 host 同步。
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
    amax = tl.load(amax_ptr)  # GPU 标量，无 host 同步
    s = tl.maximum(amax, 1e-12) / FP8_MAX  # 与 eager 的 x/(amax/448) 逐位一致（不要用倒数乘法）
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.minimum(tl.maximum(x / s, -FP8_MAX), FP8_MAX)  # 饱和防 e4m3fn 溢出成 NaN
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def _num_warps(block: int) -> int:
    return 4 if block <= 2048 else (8 if block <= 8192 else 16)


def quantize_fp8(x2: torch.Tensor):
    """x2: [M, K] contiguous bf16 → (fp8 张量, scale fp32 GPU 标量)。语义同
    s = amax/448; (x2/s).to(e4m3)，但只有 2 次读 + 1 次写。"""
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
