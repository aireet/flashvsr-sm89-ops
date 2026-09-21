"""Fused AdaLN elementwise chains (bf16, Triton).

`fused_ln_modulate`: LayerNorm (no affine) then `x * (1 + scale) + shift`.
`fused_gate_add`: `x + gate * residual`.
Each replaces 4/2 eager kernels with one; scale/shift/gate are [B, 1, D]
broadcast per row.

Numerics: replicates eager's per-op bf16 rounding chain exactly
(`bf16(1+s) -> bf16(xn*(1+s)) -> bf16(+shift)`; `bf16(g*r) -> bf16(x+·)`),
which makes `fused_gate_add` bitwise identical to eager and
`fused_ln_modulate` within 1 bf16 ulp. Do not replace the explicit
round-trips with a single fp32 rounding — parity depends on them.

Note the argument order: the wrapper takes `(scale, shift)`, the reverse of
the upstream `modulate(x, shift, scale)` call-site convention.
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
    xn = ((x - mean) / tl.sqrt(var + eps)).to(tl.bfloat16).to(tl.float32)

    b_ = row // L
    pbase = b_ * D
    s = tl.load(s_ptr + pbase + offs, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(b_ptr + pbase + offs, mask=mask, other=0.0).to(tl.float32)
    # per-op bf16 rounding, matching eager exactly
    y = (xn * (1.0 + s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    y = (y + sh).to(tl.bfloat16)
    tl.store(out_ptr + row * D + offs, y, mask=mask)


@triton.jit
def _gate_add_kernel(x_ptr, g_ptr, r_ptr, out_ptr, N, LD,
                     D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = idx < N
    # parameter offset p = batch*D + (idx % D)
    p = (idx // LD) * D + idx % D
    g = tl.load(g_ptr + p, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    y = (g * r).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + idx, (x + y).to(tl.bfloat16), mask=mask)


def fused_ln_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                      eps: float = 1e-6):
    """LayerNorm(affine=False) + modulate. x [B, L, D] contiguous bf16; scale/shift [B, 1, D]."""
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
    """x + gate * residual. x/residual [B, L, D] contiguous bf16; gate [B, 1, D]."""
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
