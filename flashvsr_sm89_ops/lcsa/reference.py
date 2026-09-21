"""PyTorch reference implementation of LCSA — the correctness oracle for the
Triton kernel.

Mirrors the upstream semantics: WindowPartition3D (win 2,8,8) makes each
128-token block a (time, row, col)-major window; block scores are window-
mean pooled, local-masked, softmaxed, then jointly top-k'd per
(head, query time row); attention runs over the selected key blocks only.
"""
import math

import torch
from einops import rearrange


def window_partition(x, win=(2, 8, 8)):
    """[B, F, H, W, C] -> [Nwin, wf*wh*ww, C]"""
    B, F, H, W, C = x.shape
    wf, wh, ww = win
    x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wf * wh * ww, C)


def build_local_block_mask(bh, bw, rng, device):
    """Sliding-window local mask, [bh*bw, bh*bw] bool (out-of-bounds -> False)."""
    r = torch.arange(bh, device=device)
    c = torch.arange(bw, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all, c_all = YY.reshape(-1), XX.reshape(-1)
    sr = r_all - rng // 2
    sc = c_all - rng // 2
    in_row = (r_all[None, :] >= sr[:, None]) & (r_all[None, :] <= sr[:, None] + rng - 1)
    in_col = (c_all[None, :] >= sc[:, None]) & (c_all[None, :] <= sc[:, None] + rng - 1)
    return in_row & in_col


def draft_block_mask(q_w, k_w, nheads, q_temporal, local, topk):
    """Block-mask selection. q_w: [Nq, 128, h*d], k_w: [Nk, 128, h*d],
    local: [sq, sk] bool, q_temporal: query time rows (f//2).
    Returns [h, Nq, Nk] bool."""
    avg_q = torch.mean(q_w, dim=1)
    avg_k = torch.mean(k_w, dim=1)
    avg_q = rearrange(avg_q, "s (h d) -> h s d", h=nheads)
    avg_k = rearrange(avg_k, "s (h d) -> h s d", h=nheads)
    D = avg_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", avg_q, avg_k) / math.sqrt(D)

    sq, sk = local.shape
    rl = scores.shape[1] // sq
    rn = scores.shape[2] // sk
    lm = local.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    lm = rearrange(lm, "x a y b -> (x a) (y b)")
    bias = torch.zeros_like(scores)
    bias = bias.masked_fill(~lm.unsqueeze(0), float("-inf"))
    scores = scores + bias

    attn = torch.softmax(scores, dim=-1)
    h, nq, nk = attn.shape
    flat = rearrange(attn, "h (t s) m -> (h t) (s m)", t=q_temporal)  # joint top-k
    apply_topk = min(flat.shape[1] - 1, topk)
    thr = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1].unsqueeze(1)
    keep = flat > thr
    keep = rearrange(keep, "(h t) (s m) -> h (t s) m", t=q_temporal, s=nq // q_temporal)
    return keep


def sparse_attention_ref(q_w, k_w, v_w, keep):
    """Exact block-sparse attention reference: per (head, query block) loop in
    fp32. q_w/k_w/v_w: [N, 128, h*d]; keep: [h, Nq, Nk] bool.
    Returns [Nq, 128, h*d] (fp32)."""
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
    """Full LCSA forward reference (no KV cache, no output projection).
    q/k/v: [1, L, h*d] (already RoPE'd), L = f*h_lat*w_lat."""
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
    local = build_local_block_mask(bh, bw, local_range, q_w.device)
    q_temporal = f // 2
    keep = draft_block_mask(q_w, k_w, nheads, q_temporal, local, topk)
    if Nk > Nq:
        # streaming case (KV cache): upstream lets cache blocks compete in the
        # same top-k; this reference only validates the Nk == Nq case.
        pass
    out = sparse_attention_ref(q_w, k_w, v_w, keep)
    return rearrange(out, "(b n) s c -> b (n s) c", b=B), keep
