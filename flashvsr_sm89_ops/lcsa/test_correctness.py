#!/usr/bin/env python3
"""LCSA 正确性与性能验证: reference(PyTorch+SDPA) vs Triton kernel。

用法: flashvsr-env/bin/python test_correctness.py [--perf]
BSA 对拍在 BSA 可用后单独脚本做（掩码格式需先对齐）。
"""
import sys
import torch

sys.path.insert(0, "/root/work/workspace/kernel-dev/kernels/lcsa")
import reference as ref
import triton_lcsa as tl_kernel

torch.manual_seed(0)
DEV = "cuda"
H, D, S = 12, 128, 128  # Wan2.1 1.3B: 12 heads × 128


def make_case(h_lat, w_lat, f=2, nk_extra=0, topk_ratio=0.25, local_range=11):
    """h_lat/w_lat: latent 尺寸（8 的倍数）。nk_extra: cache 的额外窗口时间行数。"""
    L = f * h_lat * w_lat
    bh, bw = h_lat // 8, w_lat // 8
    Nq = (f // 2) * bh * bw  # win=(2,8,8): 每 2 帧 1 个时间行
    Nk = Nq + nk_extra
    q = torch.randn(1, L, H * D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Nk * S, H * D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q_w = ref.window_partition(q.view(1, f, h_lat, w_lat, H * D))
    k_w = k.view(Nk, S, H * D)
    local = ref.build_local_block_mask(bh, bw, local_range, DEV)
    topk = int(Nq * Nk * topk_ratio)
    keep = ref.draft_block_mask(q_w, k_w, H, f // 2, local, topk)
    return q, k, v, keep


def compare(out_tri, out_ref, atol=3e-2, rtol=3e-2):
    diff = (out_tri.float() - out_ref.float()).abs()
    ok = torch.allclose(out_tri.float(), out_ref.float(), atol=atol, rtol=rtol)
    print(f"    max_abs={diff.max():.4f}  mean={diff.mean():.6f}  "
          f"allclose(atol={atol}): {ok}")
    return ok


def test_sizes():
    all_ok = True
    for (h_lat, w_lat, nk_extra, ratio) in [
        (16, 24, 0, 0.4),   # 小: Nq=6
        (32, 48, 0, 0.3),   # 中: Nq=24
        (48, 88, 3 * 66, 0.46),  # 真实: 768x1408 块网格 6x11, Nq=66, cache 3 行 (nk_extra=窗口数)
    ]:
        print(f"  case h_lat={h_lat} w_lat={w_lat} nk_extra={nk_extra} ratio={ratio}")
        q, k, v, keep = make_case(h_lat, w_lat, nk_extra=nk_extra, topk_ratio=ratio)
        f = 2
        q_w = ref.window_partition(q.view(1, f, h_lat, w_lat, H * D))
        out_ref = ref.sparse_attention_ref(q_w, k.view(-1, S, H * D), v.view(-1, S, H * D), keep)
        # kernel 的输入布局 = 窗口序（与 BSA 收到的 WindowPartition3D 输出一致）
        q_win = q_w.reshape(-1, H * D).unsqueeze(0)
        out_tri = tl_kernel.block_sparse_attention(q_win, k, v, keep)
        out_ref = out_ref.view(1, -1, H * D)
        all_ok &= compare(out_tri, out_ref, atol=1e-2, rtol=1e-2)
        del q, k, v, keep, q_w, q_win, out_ref, out_tri
        torch.cuda.empty_cache()
    return all_ok


def bench():
    print("=== perf: 768x1408 (Nq=264), bf16 ===")
    results = []
    for nk_rows, ratio in [(0, 0.5), (2, 0.3), (6, 0.25), (14, 0.15)]:
        case = make_case(96, 176, nk_extra=nk_rows, topk_ratio=ratio)
        q, k, v, keep = case
        density = keep.float().mean().item()
        # kernel 时间
        for _ in range(3):
            tl_kernel.block_sparse_attention(q, k, v, keep)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(10):
            tl_kernel.block_sparse_attention(q, k, v, keep)
        e1.record(); torch.cuda.synchronize()
        ms_attn = e0.elapsed_time(e1) / 10
        # 含 CSR 构建的真实端到端
        for _ in range(3):
            tl_kernel.build_csr(keep)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(10):
            sel, cnt, maxk = tl_kernel.build_csr(keep)
        e1.record(); torch.cuda.synchronize()
        ms_csr = e0.elapsed_time(e1) / 10
        print(f"  Nk={keep.shape[2]} density={density:.3f}: attn {ms_attn:.2f} ms "
              f"| csr {ms_csr:.2f} ms | sum {ms_attn + ms_csr:.2f} ms")
        results.append(dict(nk=int(keep.shape[2]), density=density,
                            attn_ms=ms_attn, csr_ms=ms_csr))
        del q, k, v, keep
        torch.cuda.empty_cache()
    import json
    with open("/root/work/workspace/kernel-dev/eval/lcsa_perf.json", "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    print("=== correctness ===")
    ok = test_sizes()
    print("ALL_OK" if ok else "ALL_FAIL")
    if "--perf" in sys.argv:
        bench()
