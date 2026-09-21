"""FFN with fused GELU(tanh) -> FP8 mid-section (Triton, sm_89).

Replaces `Sequential(Linear, GELU, Linear)` with
quantize -> mm0 -> [gelu+absmax] -> [gelu+quantize] -> mm2, so the bf16 GELU
output is never materialized.

Numerics: GELU runs in fp32 and is rounded to bf16 to replicate aten's
output rounding; the absmax/quantize formulas are identical to
`fp8_quant`, so codes are bitwise identical to the unfused
`quantize_fp8(F.gelu(y))` chain.

Note: Triton 3.2 has no `tl.math.tanh`; the exp identity below is used
instead. Module-level Python floats are invisible inside `@triton.jit` —
constants must be passed as `tl.constexpr` (C0, FP8_MAX).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

from flashvsr_sm89_ops.fp8_quant import FP8_MAX, quantize_fp8, _num_warps


@triton.jit
def _gelu_tanh(x, C0: tl.constexpr):
    # tanh(u) = 1 - 2/(exp(2u)+1)
    u = C0 * (x + 0.044715 * x * x * x)
    t = 1.0 - 2.0 / (tl.exp(2.0 * u) + 1.0)
    g = 0.5 * x * (1.0 + t)
    return g.to(tl.bfloat16).to(tl.float32)  # replicate aten's bf16 output rounding


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
    """y2: [M, K] contiguous bf16 -> (fp8, scale). Equivalent to
    quantize_fp8(F.gelu(y2, approximate='tanh')) without materializing the
    GELU output."""
    n = y2.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=y2.device)
    x8 = torch.empty(y2.shape, dtype=torch.float8_e4m3fn, device=y2.device)
    if n == 0:
        return x8, amax
    BLOCK = 8192 if n >= 8192 else triton.next_power_of_2(n)
    C0 = 0.7978845608028654  # sqrt(2/pi), same constant as aten tanh-GELU
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_absmax_kernel[grid](y2, amax, n, C0=C0, BLOCK=BLOCK, num_warps=_num_warps(BLOCK))
    _gelu_quant_kernel[grid](y2, amax, x8, n, C0=C0, FP8_MAX=FP8_MAX, BLOCK=BLOCK,
                             num_warps=_num_warps(BLOCK))
    return x8, amax / FP8_MAX


class FP8FFN(nn.Module):
    """Drop-in replacement for a DiT FFN `Sequential(Linear, GELU(tanh), Linear)`
    whose two Linears are already `FP8Linear`."""

    def __init__(self, lin0, lin2):
        super().__init__()
        self.lin0 = lin0
        self.lin2 = lin2

    def forward(self, x):
        from flashvsr_sm89_ops.fp8_quant import quantize_fp8
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.is_contiguous():
            x8, s = quantize_fp8(x2)
        else:
            s = (x2.abs().amax() / FP8_MAX).clamp(min=1e-12).float()
            x8 = (x2 / s).to(torch.float8_e4m3fn)
        h = self.lin0._mm(x8, s)
        h8, hs = fused_gelu_quant_fp8(h)
        out = self.lin2._mm(h8, hs)
        return out.reshape(*shape[:-1], out.shape[-1])


def convert_ffn_fp8(model: nn.Module) -> int:
    """Swap every `blocks.i.ffn` Sequential for FP8FFN; returns the count.
    Requires convert_linears_fp8(model, parts=["ffn"]) to run first."""
    import re
    pat = re.compile(r"^blocks\.\d+\.ffn$")
    n = 0
    for name, mod in list(model.named_modules()):
        if not pat.match(name):
            continue
        l0, l2 = mod[0], mod[2]
        assert type(l0).__name__ == "FP8Linear" and type(l2).__name__ == "FP8Linear", \
            f"{name}: run convert_linears_fp8(parts=['ffn']) first"
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, FP8FFN(l0, l2))
        n += 1
    return n
