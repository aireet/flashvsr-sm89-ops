#!/usr/bin/env python3
"""M4: AdaLN modulate / gate 融合 Triton kernel（wan_video_dit.py DiTBlock）。

被替换的 ATen 链（每 block 各 2 次，270 block 调用/全片）：
- modulate(norm(x)) = LN + (1+scale) + mul + add  → 4 kernel ~210MB
- x + gate*residual = mul + add                   → 2 kernel ~130MB
融合后各 1 kernel（52MB / 78MB），本段流量 -55%。

数值对齐：LN 结果复刻 reference 的 bf16 中间舍入；modulate/gate 在 fp32 完成、
单次舍入回 bf16（reference 为 2 次舍入，差 ≤1 bf16 ulp，e2e 质量门是最终裁判）。

scale/shift/gate 形状 [B,1,D]（modulation+t_mod chunk 而来），按行广播：
参数偏移 p = (idx // (L*D))*D + idx % D。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_modulate_kernel(x_ptr, s_ptr, b_ptr, out_ptr, L,
                        D: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    mean = tl.sum(x, axis=0) / D
    var = tl.sum(tl.where(mask, (x - mean) * (x - mean), 0.0), axis=0) / D
    xn = ((x - mean) / tl.sqrt(var + eps)).to(tl.bfloat16).to(tl.float32)  # 复刻 LN 输出 bf16 舍入

    b_ = row // L  # batch 索引
    pbase = b_ * D
    s = tl.load(s_ptr + pbase + offs, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(b_ptr + pbase + offs, mask=mask, other=0.0).to(tl.float32)
    # 逐位复刻 eager：bf16(1+s) → bf16(xn*(1+s)) → bf16(+shift)，每次 op 一次舍入
    y = (xn * (1.0 + s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y = (y + sh).to(tl.bfloat16)
    tl.store(out_ptr + row * D + offs, y, mask=mask)


@triton.jit
def _gate_add_kernel(x_ptr, g_ptr, r_ptr, out_ptr, N, LD,
                     D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < N
    # p = batch*D + (idx % D)；idx < 2^31 时 int64 除法仍安全
    p = (idx // LD) * D + idx % D
    g = tl.load(g_ptr + p, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    # 逐位复刻 eager：bf16(g*r) → bf16(x+·)
    y = (g * r).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + idx, (x + y).to(tl.bfloat16), mask=mask)


def fused_ln_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                      eps: float = 1e-6):
    """LayerNorm(affine=False) + modulate 融合。x [B,L,D] bf16 连续，scale/shift [B,1,D]。"""
    B, L, D = x.shape
    assert x.is_contiguous() and scale.is_contiguous() and shift.is_contiguous()
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(D)
    _ln_modulate_kernel[(B * L,)](
        x, scale, shift, out, L, D=D, eps=eps, BLOCK=BLOCK,
        num_warps=8, num_stages=1,
    )
    return out


def fused_gate_add(x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor):
    """x + gate*residual 融合。x/residual [B,L,D] bf16 连续，gate [B,1,D]。"""
    B, L, D = x.shape
    assert x.is_contiguous() and residual.is_contiguous() and gate.is_contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 4096
    _gate_add_kernel[(triton.cdiv(N, BLOCK),)](
        x, gate, residual, out, N, L * D, D=D, BLOCK=BLOCK,
        num_warps=8, num_stages=1,
    )
    return out
