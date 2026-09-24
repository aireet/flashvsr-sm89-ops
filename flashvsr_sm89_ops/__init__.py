"""flashvsr-sm89-ops — FlashVSR v1.1 operators for RTX 4090 (sm_89).

Importing this package installs a zero-compile Triton stand-in for the
mit-han-lab ``block_sparse_attn`` module when the real CUDA package is absent
(``FS89_LCSA`` controls this; see compat_bsa). Do it before importing
diffsynth, whose DiT imports block_sparse_attn at module load time.
"""
from .compat_bsa import active_backend, ensure_bsa_available

ensure_bsa_available()

from .fused_rms_rope import fused_rms_rope
from .fused_adaln import fused_ln_modulate, fused_gate_add
from .fp8_quant import quantize_fp8, FP8_MAX
from .fp8_linear import FP8Linear, convert_linears_fp8
from .fp8_ffn import FP8FFN, convert_ffn_fp8
from .integrate import enable

__all__ = [
    "enable",
    "fused_rms_rope",
    "fused_ln_modulate",
    "fused_gate_add",
    "quantize_fp8",
    "FP8_MAX",
    "FP8Linear",
    "convert_linears_fp8",
    "FP8FFN",
    "convert_ffn_fp8",
    "active_backend",
]
