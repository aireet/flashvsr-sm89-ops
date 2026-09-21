# Triton / FP8 operator pack for FlashVSR on RTX 4090 (sm_89).
# Apache-2.0. See README.md and docs/kernels.md.
#
# Modules are importable individually; the top-level names below are the
# stable public API used by the integration layer (docs/integration.md).
from .fused_rms_rope import fused_rms_rope            # RMSNorm+RoPE fusion (bf16)
from .fused_adaln import fused_ln_modulate, fused_gate_add  # AdaLN fusion (bf16)
from .fp8_quant import quantize_fp8, FP8_MAX           # fused absmax+quant (Triton)
from .fp8_linear import FP8Linear, convert_linears_fp8 # E4M3 per-tensor Linear via torch._scaled_mm
from .fp8_ffn import FP8FFN, convert_ffn_fp8           # GELU(tanh)->FP8 fused mid-section

__all__ = [
    "fused_rms_rope",
    "fused_ln_modulate",
    "fused_gate_add",
    "quantize_fp8",
    "FP8_MAX",
    "FP8Linear",
    "convert_linears_fp8",
    "FP8FFN",
    "convert_ffn_fp8",
]
