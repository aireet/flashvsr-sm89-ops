"""Zero-compile stand-in for mit-han-lab's ``block_sparse_attn`` on sm_89.

Upstream FlashVSR does ``from block_sparse_attn import block_sparse_attn_func``
at the top of its DiT module, so without that package installed — and building
it on a 4090 means a long nvcc run with the arch-80 SASS trick — the import
fails before any operator can be wired in.

Importing ``flashvsr_sm89_ops`` calls :func:`ensure_bsa_available`, which
installs a Triton JIT equivalent into ``sys.modules`` **only when the real
package is absent**:

- real BSA installed -> untouched, the CUDA kernel is used
- otherwise          -> our Triton LCSA kernel serves the call (parity within
  noise, see benchmarks/bsa_compare.json)

``FS89_LCSA`` selects the path when both would qualify: ``auto`` (default,
stub only if real package missing), ``triton`` (always use the stub) or
``bsa`` (never stub; the real package must be installed).
"""
import os
import sys
import types

import torch

STUB_VERSION = "flashvsr-sm89-ops/triton-stub"

_WARNED = False


def _triton_block_sparse_attn_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    attn_mask,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
    """Drop-in for the one call shape FlashVSR uses.

    q/k/v: [S, H, D] bf16 varlen (batch 1) in window order; attn_mask:
    [1, H, Nq, Nk] bool block mask. Returns [S, H, D] like the CUDA kernel.
    """
    global _WARNED
    if streaming_info is not None or is_causal or dropout_p != 0.0 or return_attn_probs:
        raise NotImplementedError(
            "flashvsr-sm89-ops Triton LCSA covers the FlashVSR call shape only "
            "(dense-per-batch, no dropout, no causal, no attn-probs). "
            "Install the real block_sparse_attn package for other shapes.")

    from .lcsa.triton_lcsa import block_sparse_attention

    sq, heads, dim = q.shape
    keep = attn_mask
    if keep.dim() == 4:
        keep = keep[0]
    q4 = q.reshape(1, sq, heads * dim)
    k4 = k.reshape(1, k.shape[0], heads * dim)
    v4 = v.reshape(1, v.shape[0], heads * dim)
    out = block_sparse_attention(q4, k4, v4, keep, sm_scale=softmax_scale)
    return out.reshape(sq, heads, dim)


def ensure_bsa_available():
    """Install the Triton stub unless the real package should win."""
    mode = os.environ.get("FS89_LCSA", "auto").lower()
    present = "block_sparse_attn" in sys.modules
    if not present:
        try:
            import block_sparse_attn  # noqa: F401
            present = True
        except ImportError:
            present = False

    if mode == "bsa":
        return "bsa"
    if present and mode == "auto":
        return "bsa"

    stub = types.ModuleType("block_sparse_attn")
    stub.block_sparse_attn_func = _triton_block_sparse_attn_func
    stub.__version__ = STUB_VERSION
    sys.modules["block_sparse_attn"] = stub
    return "triton"


def active_backend():
    """Report which LCSA backend ``import block_sparse_attn`` resolves to."""
    mod = sys.modules.get("block_sparse_attn")
    if mod is None:
        return "bsa" if os.environ.get("FS89_LCSA", "auto").lower() == "bsa" else "unresolved"
    return "triton" if getattr(mod, "__version__", "") == STUB_VERSION else "bsa"
