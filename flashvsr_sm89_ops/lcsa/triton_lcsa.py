#!/usr/bin/env python3
"""M1: Triton 块稀疏注意力（4090 / sm_89 原生）。

与官方路径的关键差异：
- 输入 = 窗口序 [1, N_win*128, H*D]（WindowPartition3D 之后的布局，与 BSA API 对齐）；
  注意：窗口的 128 token 在栅格序里是 2帧×8段×8token 的跨步散布，**不等于**连续 128 块，
  所以"免 reorder 直读栅格序"要靠把 (nq,s)→raster 的索引算术折进寻址——那是下一步的优化项
- 掩码以 CSR (SEL_IDX/SEL_CNT) 传入，不物化 [h, Lq, Lk] token 级掩码
- sm_80 二进制兼容的 BSA 在 Ada 上不是最优；本 kernel 用 Ada 的完整 smem/寄存器调度
"""
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
    tok = nq * BLOCK_S + offs_s  # 窗口在 token 序里连续 => 直接索引

    HD = NUM_HEADS * D
    q = tl.load(Q + tok[:, None] * HD + h * D + offs_d[None, :])  # [S, D] bf16

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

    # cnt==0 的行保持 0（不产生 NaN）；与 CSR 参照一致
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(O + tok[:, None] * HD + h * D + offs_d[None, :], acc.to(O.dtype.element_ty))


def build_csr(keep: torch.Tensor):
    """keep [H, NQ, NK] bool -> (sel_idx [H,NQ,MAXK] i32, cnt [H,NQ] i32, MAXK)"""
    H, NQ, NK = keep.shape
    cnt = keep.sum(-1)
    maxk = max(int(cnt.max()), 1)
    order = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True)
    sel = order[..., :maxk].to(torch.int32).contiguous()
    return sel, cnt.to(torch.int32).contiguous(), maxk


def block_sparse_attention(q, k, v, keep, sm_scale=None):
    """q/k/v: [1, L, H*D] bf16（token 序 = 窗口序）; keep: [H, L//128, K//128] bool。

    返回 [1, L, H*D] bf16。注意：cnt==0 的行输出 0（对照 BSA 行为需在测试中确认）。
    """
    L, HD = q.shape[-2], q.shape[-1]
    H = keep.shape[0]
    D = HD // H
    NQ, NK = keep.shape[1], keep.shape[2]
    assert L == NQ * 128 and k.shape[1] == NK * 128
    if sm_scale is None:
        sm_scale = 1.0 / math_sqrt(D)
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


def math_sqrt(x):
    import math
    return math.sqrt(x)
