#!/usr/bin/env python3
"""二分定位: kernel vs torch-CSR(同 keep, fp32 逐块 gather) —— 区分 kernel bug / reference bug。"""
import sys, math
import torch
from einops import rearrange
sys.path.insert(0, "/root/work/workspace/kernel-dev/kernels/lcsa")
import reference as ref
import triton_lcsa as tlk

torch.manual_seed(0)
H, D, S = 12, 128, 128
DEV = "cuda"

h_lat, w_lat, ratio = 16, 24, 0.4
f = 2
L = f * h_lat * w_lat
Nq = L // S
bh, bw = h_lat // 8, w_lat // 8
Nk = Nq

q = torch.randn(1, L, H * D, device=DEV, dtype=torch.bfloat16)
k = torch.randn(1, Nk * S, H * D, device=DEV, dtype=torch.bfloat16)
v = torch.randn_like(k)

q_w = ref.window_partition(q.view(1, f, h_lat, w_lat, H * D))
k_w = k.view(Nk, S, H * D)
v_w = v.view(Nk, S, H * D)
local = ref.build_local_block_mask(bh, bw, 11, DEV)
keep = ref.draft_block_mask(q_w, k_w, H, f // 2, local, int(Nq * Nk * ratio))

print("keep density:", keep.float().mean().item(),
      "| zero rows per head:", (keep.sum(-1) == 0).sum(-1).tolist())

# --- Triton kernel 输出（喂窗口序：与 BSA 收到的布局一致）
q_win = q_w.reshape(Nq * S, H * D).unsqueeze(0)
k_win = k_w.reshape(Nk * S, H * D).unsqueeze(0)
v_win = v_w.reshape(Nk * S, H * D).unsqueeze(0)
out_tri = tlk.block_sparse_attention(q_win, k_win, v_win, keep).view(Nq, S, H * D)

# --- torch CSR 逐块 fp32（按 head 切分，与 reference 同语义）
sel, cnt, maxk = tlk.build_csr(keep)
q_hp = rearrange(q_w, "n s (h d) -> h n s d", h=H).float()
k_hp = rearrange(k_w, "n s (h d) -> h n s d", h=H).float()
v_hp = rearrange(v_w, "n s (h d) -> h n s d", h=H).float()
out_t = torch.zeros(H, Nq, S, D, device=DEV)
for hi in range(H):
    for nq in range(Nq):
        c = int(cnt[hi, nq])
        if c == 0:
            continue
        idxs = sel[hi, nq, :c].long()
        kk = k_hp[hi, idxs].reshape(-1, D)
        vv = v_hp[hi, idxs].reshape(-1, D)
        sc = q_hp[hi, nq] @ kk.T / math.sqrt(D)
        p = torch.softmax(sc, -1)
        out_t[hi, nq] = p @ vv
out_t = rearrange(out_t, "h n s d -> n s (h d)")

d = (out_tri.float() - out_t.float()).abs()
print(f"kernel vs torchCSR: max={d.max():.4f} mean={d.mean():.6f}")

# --- reference (金标准) vs torchCSR —— 定位 reference 是否有 bug
out_ref = ref.sparse_attention_ref(q_w, k_w, v_w.view(Nk, S, H * D), keep)
d2 = (out_ref.to(torch.bfloat16).float() - out_t.float()).abs()
print(f"ref vs torchCSR:    max={d2.max():.4f} mean={d2.mean():.6f}")

# --- keep 掩码与 official 的差（无 BSA 时跳过）
try:
    sys.path.insert(0, "/root/work/workspace/kernel-dev/flashvsr")
    from diffsynth.models.wan_video_dit import generate_draft_block_mask, build_local_block_mask_shifted_vec_normal_slide
    loc_off = build_local_block_mask_shifted_vec_normal_slide(bh, bw, 11, 11, include_self=True, device=DEV)
    keep_off = generate_draft_block_mask(1, H, f // 2, q_w, k_w, topk=int(Nq * Nk * ratio), local_attn_mask=loc_off)
    keep_off = keep_off[0]
    same = (keep_off == keep)
    print(f"my keep vs official keep: 一致率={same.float().mean():.6f} | 完全一致={bool(same.all())}")
except Exception as e:
    print("official keep 对比跳过:", type(e).__name__, str(e)[:120])
