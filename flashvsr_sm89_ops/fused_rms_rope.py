"""Fused RMSNorm + RoPE (bf16, Triton).

Replaces a 5-kernel RMSNorm chain plus a complex64/128 RoPE apply with one
kernel: one program per row, fp32 reduction, RoPE pair rotation via an
`offs ^ 1` partner-lane load (same 128B cache line, L1-resident).

Numerics: RMS reduction in fp32; the normalized value is rounded to bf16
before the weight multiply to replicate eager's intermediate rounding.
RoPE rotation itself runs in fp32 (reference uses fp64 complex; the gap,
~1e-7, is far below bf16 quantization).

`freqs_cis` is the complex tensor with tail dim D//2 ([S, 1, 64] in
FlashVSR); cos/sin are shared across heads. With `rope=False` the kernel
degrades to plain RMSNorm + weight (e.g. cross-attention q/k norms).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_rope_kernel(
    x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
    S, D: tl.constexpr, HALF: tl.constexpr, HEAD_DIM: tl.constexpr,
    eps: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    s = row % S
    base = row.to(tl.int64) * D
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    rrms = tl.rsqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # bf16 round-trip replicates eager's intermediate rounding
    y = (x * rrms).to(tl.bfloat16).to(tl.float32) * w

    if ROPE:
        o = offs % HEAD_DIM
        i = o // 2
        odd = (o % 2) == 1
        # partner lane via offs^1: same cache line, so the reload hits L1
        xp = tl.load(x_ptr + base + (offs ^ 1), mask=mask, other=0.0).to(tl.float32)
        wp = tl.load(w_ptr + (offs ^ 1), mask=mask, other=0.0).to(tl.float32)
        yp = (xp * rrms).to(tl.bfloat16).to(tl.float32) * wp
        fbase = s.to(tl.int64) * HALF
        c = tl.load(cos_ptr + fbase + i, mask=mask, other=0.0)
        sn = tl.load(sin_ptr + fbase + i, mask=mask, other=0.0)
        # even lane: y_e*c - y_o*s ; odd lane: y_e*s + y_o*c
        out = y * c + yp * tl.where(odd, sn, -sn)
        tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + base + offs, y.to(tl.bfloat16), mask=mask)


def _freqs_cos_sin(freqs: torch.Tensor):
    """[..., S, 64] complex -> (cos [S, 64] fp32 contiguous, sin likewise)."""
    fr = torch.view_as_real(freqs.reshape(-1, freqs.shape[-1]))
    cos = fr[..., 0].float().contiguous()
    sin = fr[..., 1].float().contiguous()
    return cos, sin


def fused_rms_rope(x: torch.Tensor, weight: torch.Tensor, freqs_cis, eps: float = 1e-6,
                   rope: bool = True, head_dim: int = 128):
    """x [B, L, D] bf16 -> RMSNorm(weight) -> [x RoPE]."""
    B, L, D = x.shape
    x2 = x.reshape(B * L, D)
    out = torch.empty_like(x2)
    if rope:
        cos, sin = _freqs_cos_sin(freqs_cis)  # ~0.1ms per call; cache at call site if hot
    else:
        cos = torch.empty(0, device=x.device)
        sin = cos
    BLOCK = triton.next_power_of_2(D)
    _rms_rope_kernel[(B * L,)](
        x2, weight, cos, sin, out, L,
        D=D, HALF=head_dim // 2, HEAD_DIM=head_dim, eps=eps,
        ROPE=rope, BLOCK=BLOCK, num_warps=8, num_stages=1,
    )
    return out.reshape(B, L, D)
