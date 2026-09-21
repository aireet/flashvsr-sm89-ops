#!/usr/bin/env python3
"""LCSA 的 PyTorch 参照实现 —— 严格复刻 wan_video_dit.py 的语义，作为 Triton kernel 的正确性金标准。

语义锚点（flashvsr/diffsynth/models/wan_video_dit.py）:
- WindowPartition3D (win 2,8,8) → 每 128 token 一块, 窗口顺序 = (时间, 行, 列) major-to-minor
- generate_draft_block_mask: 块均值池化 → [h,Lq,Lk] 分数 → local 滑窗 -inf → softmax
  → 按 (head, 查询时间行) 对 (spatial_win × 全部 key 块) 联合 top-k → 布尔块掩码
- 注意力: 对每块 128 token, 只在选中的 key 块上做 softmax(QK^T/sqrt(d))V
"""
import math
import torch
from einops import rearrange


def window_partition(x, win=(2, 8, 8)):
    """[B,F,H,W,C] -> [Nwin, wf*wh*ww, C]"""
    B, F, H, W, C = x.shape
    wf, wh, ww = win
    x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wf * wh * ww, C)


def build_local_block_mask(bh, bw, rng, device):
    """normal_slide 版本（无 clamp，出界自然为 False）—— 复刻 build_local_block_mask_shifted_vec_normal_slide"""
    r = torch.arange(bh, device=device)
    c = torch.arange(bw, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all, c_all = YY.reshape(-1), XX.reshape(-1)
    sr = r_all - rng // 2
    sc = c_all - rng // 2
    in_row = (r_all[None, :] >= sr[:, None]) & (r_all[None, :] <= sr[:, None] + rng - 1)
    in_col = (c_all[None, :] >= sc[:, None]) & (c_all[None, :] <= sc[:, None] + rng - 1)
    return in_row & in_col  # [bh*bw, bh*bw]


def draft_block_mask(q_w, k_w, nheads, q_temporal, local, topk):
    """复刻 generate_draft_block_mask（不含 @no_grad 与 batch 维）。

    q_w: [Nq, 128, h*d]  k_w: [Nk, 128, h*d]  local: [sq, sk] bool
    q_temporal: 查询窗口的时间行数（f//2）；返回 [h, Nq, Nk] bool
    """
    avg_q = torch.mean(q_w, dim=1)
    avg_k = torch.mean(k_w, dim=1)
    avg_q = rearrange(avg_q, "s (h d) -> h s d", h=nheads)
    avg_k = rearrange(avg_k, "s (h d) -> h s d", h=nheads)
    D = avg_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", avg_q, avg_k) / math.sqrt(D)  # [h, Nq, Nk]

    sq, sk = local.shape
    rl = scores.shape[1] // sq
    rn = scores.shape[2] // sk
    lm = local.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    lm = rearrange(lm, "x a y b -> (x a) (y b)")
    bias = torch.zeros_like(scores)
    bias = bias.masked_fill(~lm.unsqueeze(0), float("-inf"))
    scores = scores + bias

    attn = torch.softmax(scores, dim=-1)  # [h, Nq, Nk]
    h, nq, nk = attn.shape
    flat = rearrange(attn, "h (t s) m -> (h t) (s m)", t=q_temporal)  # 联合 top-k
    apply_topk = min(flat.shape[1] - 1, topk)
    thr = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1].unsqueeze(1)
    keep = flat > thr
    keep = rearrange(keep, "(h t) (s m) -> h (t s) m", t=q_temporal, s=nq // q_temporal)
    return keep


def sparse_attention_ref(q_w, k_w, v_w, keep):
    """块稀疏注意力精确参照：逐 (head, query块) 循环，fp32 计算，显存 O(块级)。

    q_w/k_w/v_w: [N, 128, h*d]；keep: [h, Nq, Nk] bool。返回 [Nq, 128, h*d]（fp32）。
    """
    Nq, S, HD = q_w.shape
    h = keep.shape[0]
    d = HD // h
    scale = 1.0 / math.sqrt(d)

    q = rearrange(q_w, "n s (hd d) -> hd n s d", hd=h).float()
    k = rearrange(k_w, "n s (hd d) -> hd n s d", hd=h).float()
    v = rearrange(v_w, "n s (hd d) -> hd n s d", hd=h).float()
    out = torch.zeros_like(q)
    for hi in range(h):
        for nq in range(Nq):
            sel = keep[hi, nq].nonzero(as_tuple=True)[0]
            kk = k[hi, sel].reshape(-1, d)
            vv = v[hi, sel].reshape(-1, d)
            sc = q[hi, nq] @ kk.T * scale
            p = torch.softmax(sc, dim=-1)
            out[hi, nq] = p @ vv
    return rearrange(out, "hd n s d -> n s (hd d)", hd=h)


def lcsa_forward_ref(q, k, v, f, h_lat, w_lat, nheads, topk, local_range=11):
    """完整 LCSA 前向参照（等价 self.attn(reorder...) 路径，不含 KV cache 与 o 投影）。

    q/k/v: [1, L, h*d]（已 RoPE），L = f*h_lat*w_lat
    """
    B, L, HD = q.shape
    xq = q.view(B, f, h_lat, w_lat, HD)
    xk = k.view(B, f, h_lat, w_lat, HD)
    xv = v.view(B, f, h_lat, w_lat, HD)
    q_w = window_partition(xq)
    k_w = window_partition(xk)
    v_w = window_partition(xv)
    Nq = q_w.shape[0]
    Nk = k_w.shape[0]

    bh, bw = h_lat // 8, w_lat // 8
    local = build_local_block_mask(bh, bw, local_range, q_w.device)  # [bh*bw, bh*bw]
    q_temporal = f // 2
    keep = draft_block_mask(q_w, k_w, nheads, q_temporal, local, topk)
    if Nk > Nq:  # 有 KV cache（流式）：把 keep 在 key 维左侧补 True（cache 部分全保留不合适——
        pass     # 官方实现里 cache 块参与同一 top-k；参照实现仅在 Nk==Nq 时校验）
    out = sparse_attention_ref(q_w, k_w, v_w, keep)
    return rearrange(out, "(b n) s c -> b (n s) c", b=B), keep
