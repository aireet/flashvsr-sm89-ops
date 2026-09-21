#!/usr/bin/env python3
"""M4: RMSNorm+RoPE 融合 Triton kernel（替代 fp64 复数 rope + 5-kernel RMSNorm 链）。

背景（wan_video_dit.py）：SelfAttention 对 q/k 各做 norm_q/k（pow/mean/rsqrt/mul/mul
5 kernel）→ rope_apply（to fp64 → view_as_complex → complex mul → to bf16，4 kernel，
~370MB 流量/次）。540 个 rope 实例 + 810 个 norm 实例/全片。

本 kernel：一行 [D] 一个 program，norm 归约 fp32 一次读完成，pair 旋转用 offs^1 重读
（同 cache line，L1 命中）。数学与 reference 对齐：
- RMSNorm: x.float() 归约 → norm 结果 cast 回 bf16 再乘 weight（复刻中间舍入）
- rope 旋转 fp32（reference fp64 complex，差 ~1e-7 ≪ bf16 量化 8e-3）
- ROPE=False 退化为纯 RMSNorm+weight（覆盖 cross_attn.norm_q 等无 rope 场景）

freqs：[S,64] fp32 的 cos/sin（wrapper 从 [S,1,64] complex128 拆出），所有 head 共享。
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
    y = (x * rrms).to(tl.bfloat16).to(tl.float32) * w  # 复刻 reference 的 bf16 中间舍入

    if ROPE:
        o = offs % HEAD_DIM        # head 内偏移
        i = o // 2                 # pair 索引 ∈ [0, HALF)
        odd = (o % 2) == 1
        # 配对 lane（offs^1）：同 cache line，L1 命中；norm/weight 用同一 rrms 重建
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
    """[S,1,64] complex（任意 ndim，尾维 64）→ (cos [S,64] fp32 contiguous, sin 同)。"""
    fr = torch.view_as_real(freqs.reshape(-1, freqs.shape[-1]))  # [S,64,2] fp64
    cos = fr[..., 0].float().contiguous()
    sin = fr[..., 1].float().contiguous()
    return cos, sin


def fused_rms_rope(x: torch.Tensor, weight: torch.Tensor, freqs_cis, eps: float = 1e-6,
                   rope: bool = True, head_dim: int = 128):
    """x [B, L, D] bf16 → RMSNorm(weight) → [×RoPE]；freqs_cis 为 complex 尾维 64 张量。"""
    B, L, D = x.shape
    x2 = x.reshape(B * L, D)
    out = torch.empty_like(x2)
    if rope:
        cos, sin = _freqs_cos_sin(freqs_cis)  # 每次拆换约 0.1ms，可由调用方缓存
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
