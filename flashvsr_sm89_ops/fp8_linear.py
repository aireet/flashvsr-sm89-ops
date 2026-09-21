"""FP8 (E4M3, per-tensor scale) Linear via torch._scaled_mm (cuBLASLt, sm_89).

Weights are quantized statically at conversion time; activations get a
dynamic per-tensor scale on every forward. Output is bf16. A custom Triton
FP8 GEMM measured slower than cuBLASLt across this pipeline's shapes, so
`torch._scaled_mm` is kept as the backend.
"""
import re

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
        else:
            s = (x2.abs().amax() / FP8_MAX).clamp(min=1e-12).float()
            x8 = (x2 / s).to(torch.float8_e4m3fn)
        y = self._mm(x8, s)
        return y.reshape(*shape[:-1], y.shape[-1])


def convert_linears_fp8(model: nn.Module, parts=("ffn", "self", "cross")) -> int:
    """Convert DiT-block Linears whose module names match `parts`
    ("ffn" / "self" / "cross"); returns the number converted. Modules outside
    `blocks.*` are left untouched."""
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
        setattr(parent, leaf, FP8Linear(mod))
        n += 1
    return n
