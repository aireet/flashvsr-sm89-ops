"""Triton block-sparse attention (online softmax) for LCSA on sm_89.

Layout contract: q/k/v arrive in **window order** [1, N_win*128, H*D] — the
output of WindowPartition3D. The 128 tokens of a window are strided across
raster order, not contiguous; the kernel indexes tokens in window order
throughout, so callers must not reorder.

Masks arrive in CSR form (sel_idx/sel_cnt per (head, query block)) instead
of a materialized [H, Lq, Lk] boolean mask. Rows with an empty selection
output zeros (no NaN), matching the CSR reference.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _bsa_fwd_kernel(
    Q, K, V, O,
    SEL_IDX, SEL_CNT,
    sm_scale,
    NQ,
    MAXK,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    nq = pid % NQ
    h = pid // NQ

    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    tok = nq * BLOCK_S + offs_s  # query tokens of a window are contiguous in window order

    HD = NUM_HEADS * D
    q = tl.load(Q + tok[:, None] * HD + h * D + offs_d[None, :])

    cnt = tl.load(SEL_CNT + h * NQ + nq)
    m_i = tl.full([BLOCK_S], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_S], tl.float32)
    acc = tl.zeros([BLOCK_S, D], tl.float32)

    for j in range(cnt):
        nk = tl.load(SEL_IDX + (h * NQ + nq) * MAXK + j)
        ktok = nk * BLOCK_S + offs_s
        k = tl.load(K + ktok[:, None] * HD + h * D + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(V.dtype.element_ty), tl.load(
            V + ktok[:, None] * HD + h * D + offs_d[None, :]))
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)  # empty rows stay 0 instead of NaN
    acc = acc / l_safe[:, None]
    tl.store(O + tok[:, None] * HD + h * D + offs_d[None, :], acc.to(O.dtype.element_ty))


def build_csr(keep: torch.Tensor):
    """keep [H, NQ, NK] bool -> (sel_idx [H, NQ, MAXK] i32, cnt [H, NQ] i32, MAXK)"""
    H, NQ, NK = keep.shape
    cnt = keep.sum(-1)
    maxk = max(int(cnt.max()), 1)
    order = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True)
    sel = order[..., :maxk].to(torch.int32).contiguous()
    return sel, cnt.to(torch.int32).contiguous(), maxk


def block_sparse_attention(q, k, v, keep, sm_scale=None):
    """q/k/v: [1, L, H*D] bf16 in window order; keep: [H, L//128, K//128] bool.
    Returns [1, L, H*D] bf16. Rows with no selected key blocks output 0."""
    L, HD = q.shape[-2], q.shape[-1]
    H = keep.shape[0]
    D = HD // H
    NQ, NK = keep.shape[1], keep.shape[2]
    assert L == NQ * 128 and k.shape[1] == NK * 128
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sel, cnt, maxk = build_csr(keep)
    out = torch.empty_like(q)
    grid = (H * NQ,)
    _bsa_fwd_kernel[grid](
        q, k, v, out, sel, cnt,
        sm_scale,
        NQ, maxk,
        NUM_HEADS=H, D=D, BLOCK_S=128,
        num_warps=8, num_stages=2,
    )
    return out
