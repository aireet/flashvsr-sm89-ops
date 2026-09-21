#!/usr/bin/env python3
"""M4 Kernel #4: FFN 中段 GELU→FP8 融合（fp8_ffn：quant→mm0→[gelu+amax]→[gelu+quant]→mm2）。

流量账（M=55296，bf16=2B，中段张量 992MB）：
  原链 mm0写992 | gelu 读+写 2×992 | quant 读992 写496 | mm2 读496 = 5.45GB
  融合 mm0写992 | gelu+amax 读992 | gelu+quant 读992 写496 | mm2 读496 = 3.47GB
  M=18432（331MB 张量，240 次/全片）+ M=55296（30 次）→ 推算省 ~220ms/全片。

数值：gelu(tanh) 在 fp32 计算后复刻 aten 的 bf16 输出舍入，amax/quant 与
fp8_quant.py 完全同式 → 与"先 gelu 再 quantize_fp8"的旧链只差 fp32 舍入序（≤1 code）。
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

from flashvsr_sm89_ops.fp8_quant import FP8_MAX, quantize_fp8, _num_warps

@triton.jit
def _gelu_tanh(x, C0: tl.constexpr):
    # tanh(u) = 1 - 2/(exp(2u)+1)（triton 3.2 无 tl.math.tanh）；fp32 与 aten 差 ~1 ulp，
    # bf16 舍入后与 aten gelu(tanh) 对拍无感（见 eval/fp8_ffn_bench.json）
    u = C0 * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)
    g = 0.5 * x * (1.0 + t)
    return g.to(tl.bfloat16).to(tl.float32)  # 复刻 aten gelu(bf16 输出) 的舍入


@triton.jit
def _gelu_absmax_kernel(x_ptr, amax_ptr, n, C0: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = _gelu_tanh(x, C0)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(g), axis=0))


@triton.jit
def _gelu_quant_kernel(x_ptr, amax_ptr, out_ptr, n, C0: tl.constexpr,
                       FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    amax = tl.load(amax_ptr)
    s = tl.maximum(amax, 1e-12) / FP8_MAX
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = _gelu_tanh(x, C0)
    v = tl.minimum(tl.maximum(g / s, -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def fused_gelu_quant_fp8(y2: torch.Tensor):
    """y2: [M, K] contiguous bf16（GEMM0 输出）→ (fp8, scale fp32 GPU 标量)。
    语义同 quantize_fp8(F.gelu(y2, approximate='tanh'))，省 gelu 的物化往返。"""
    n = y2.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=y2.device)
    x8 = torch.empty(y2.shape, dtype=torch.float8_e4m3fn, device=y2.device)
    if n == 0:
        return x8, amax
    BLOCK = 8192 if n >= 8192 else triton.next_power_of_2(n)
    C0 = 0.7978845608028654  # sqrt(2/pi)，aten tanh-gelu 同款
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_absmax_kernel[grid](y2, amax, n, C0=C0, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    _gelu_quant_kernel[grid](y2, amax, x8, n, C0=C0, FP8_MAX=FP8_MAX, BLOCK=BLOCK,
                             num_warps=_num_warps(BLOCK))
    return x8, amax / FP8_MAX


class FP8FFN(nn.Module):
    """DiTBlock.ffn = Sequential(Linear, GELU(tanh), Linear) 的整体 FP8 替身。

    中段不物化 bf16 gelu：mm0 输出直接走 [gelu+amax]→[gelu+quant] 两个 pass 进 fp8。
    """

    def __init__(self, lin0, lin2):
        super().__init__()
        self.lin0 = lin0  # FP8Linear
        self.lin2 = lin2  # FP8Linear

    def forward(self, x):
        from flashvsr_sm89_ops.fp8_quant import quantize_fp8
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.is_contiguous():
            x8, s = quantize_fp8(x2)
        else:
            s = (x2.abs().amax() / FP8_MAX).clamp(min=1e-12).float()
            x8 = (x2 / s).to(torch.float8_e4m3fn)
        h = self.lin0._mm(x8, s)                # bf16 [M, 8960]
        h8, hs = fused_gelu_quant_fp8(h)
        out = self.lin2._mm(h8, hs)
        return out.reshape(*shape[:-1], out.shape[-1])


def convert_ffn_fp8(model: nn.Module) -> int:
    """把 blocks.i.ffn（含 FP8Linear 的 Sequential）整体换成 FP8FFN。返回转换数。"""
    import re
    pat = re.compile(r"^blocks\.\d+\.ffn$")
    n = 0
    for name, mod in list(model.named_modules()):
        if not pat.match(name):
            continue
        l0, l2 = mod[0], mod[2]
        assert type(l0).__name__ == "FP8Linear" and type(l2).__name__ == "FP8Linear", \
            f"{name}: 需先经 convert_linears_fp8(parts=['ffn']) 转换"
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, FP8FFN(l0, l2))
        n += 1
    return n
