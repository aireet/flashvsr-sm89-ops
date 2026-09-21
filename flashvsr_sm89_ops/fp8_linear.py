#!/usr/bin/env python3
"""M3: DiT Linear 层 FP8 化（E4M3 + per-tensor dynamic scale, torch._scaled_mm / sm89）。

- 权重：加载时静态量化 scale_w = amax(|W|)/448，存 fp8
- 激活：每次 forward 动态 per-tensor scale（一次 amax 读 pass，对 ms 级 GEMM 可忽略）
- 输出 bf16；norm/attention/非目标层不动
- 灰度开关：convert_linears_fp8(model, parts={"ffn","self","cross"})，默认全开

⚠️ 质量门禁是最终裁决（eval/quality_ref.py cmp），本模块只保证数值路径正确。
"""
import torch
import torch.nn as nn

FP8_MAX = 448.0  # e4m3fn max


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.detach()
        self.w_scale = (w.abs().amax() / FP8_MAX).clamp(min=1e-12).float()
        self.weight = nn.Parameter(
            (w / self.w_scale).to(torch.float8_e4m3fn), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    def _mm(self, x8, s):
        return torch._scaled_mm(x8, self.weight.t(), scale_a=s, scale_b=self.w_scale,
                                bias=self.bias, out_dtype=torch.bfloat16)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.is_contiguous():
            from flashvsr_sm89_ops.fp8_quant import quantize_fp8
            x8, s = quantize_fp8(x2)
        else:  # 兜底（罕见）：reshape 已复制的场景不会走到这
            s = (x2.abs().amax() / FP8_MAX).clamp(min=1e-12).float()
            x8 = (x2 / s).to(torch.float8_e4m3fn)
        y = self._mm(x8, s)
        return y.reshape(*shape[:-1], y.shape[-1])


def _convert(linear: nn.Linear) -> FP8Linear:
    new = FP8Linear(linear)
    return new


def convert_linears_fp8(model: nn.Module, parts=("ffn", "self", "cross")) -> int:
    """按部件名灰度转换 DiT blocks 内的 Linear；返回转换数量。非 blocks 内的不动。"""
    import re
    allowed = []
    if "self" in parts:
        allowed += [r"^blocks\.\d+\.self_attn\.[qkvo]$"]
    if "cross" in parts:
        allowed += [r"^blocks\.\d+\.cross_attn\.[qkvo]$"]
    if "ffn" in parts:
        allowed += [r"^blocks\.\d+\.ffn\.[02]$"]
    allowed = [re.compile(p) for p in allowed]
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(p.match(name) for p in allowed):
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, _convert(mod))
        n += 1
    return n
