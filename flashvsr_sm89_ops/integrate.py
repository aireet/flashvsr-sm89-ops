"""One-call wiring of every operator into a FlashVSR v1.1 Tiny pipeline.

``enable(pipe)`` is called once, after ``pipe.to('cuda')`` /
``pipe.enable_vram_management(...)`` and before ``pipe.init_cross_kv()``.
It applies, in dependency order:

1. TCDecoder -> channels_last (bitwise-identical, ~ -17% decode time)
2. optional MemBlock ``torch.compile`` (off by default: +0.95 GiB, slow warmup)
3. DiT Linears -> FP8 E4M3 (cuBLASLt) + fused GELU FP8 FFN mid-section
4. fused RMSNorm+RoPE and fused AdaLN, by rebinding the DiT module's
   ``SelfAttention.forward`` / ``CrossAttention.forward`` / ``DiTBlock.forward``
   with fused-aware copies (the upstream stock path is kept verbatim as the
   fallback branch, so every flag can be turned off without re-cloning)
5. the block-sparse attention backend resolves through ``compat_bsa`` — real
   CUDA BSA if installed, otherwise the Triton LCSA kernel (no compilation)

Every step is env-flag controllable for A/B (see docs/integration.md).
Re-running ``enable`` on an already-patched pipeline is safe.
"""
import os

import torch

from .compat_bsa import active_backend, ensure_bsa_available
from .fp8_ffn import convert_ffn_fp8
from .fp8_linear import convert_linears_fp8
from .fused_adaln import fused_gate_add, fused_ln_modulate
from .fused_rms_rope import fused_rms_rope

_FLAG = lambda name, default: os.environ.get(name, default) == "1"


def _unwrap(m):
    """enable_vram_management wraps norms/linears in AutoWrappedModule."""
    return getattr(m, "module", m)


def _module_eps(m, default=1e-6):
    return getattr(_unwrap(m), "eps", default)


def _make_self_attn_forward(dit):
    """Fused-aware copy of upstream SelfAttention.forward; the stock branch is
    kept verbatim so FUSED_ROPE_FN=None reproduces upstream exactly."""
    from einops import rearrange

    def forward(self, x, freqs, f=None, h=None, w=None, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range=9):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f == 2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f == 6, " start f must be 6"
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        rope_fn = getattr(dit, "FUSED_ROPE_FN", None)
        if rope_fn is not None:
            q = rope_fn(self.q(x), _unwrap(self.norm_q).weight, freqs,
                        eps=_module_eps(self.norm_q), rope=True, head_dim=self.head_dim)
            k = rope_fn(self.k(x), _unwrap(self.norm_k).weight, freqs,
                        eps=_module_eps(self.norm_k), rope=True, head_dim=self.head_dim)
        else:
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            q = dit.rope_apply(q, freqs, self.num_heads)
            k = dit.rope_apply(k, freqs, self.num_heads)
        v = self.v(x)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = dit.WindowPartition3D.partition(q, win)
        k_w = dit.WindowPartition3D.partition(k, win)
        v_w = dit.WindowPartition3D.partition(v, win)

        seqlen = f // win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                              block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                              block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                              block_n=block_n_kv, block_s=block_s)

        if self.local_attn_mask is None or self.local_attn_mask_h != h // 8 \
                or self.local_attn_mask_w != w // 8 or self.local_range != local_range:
            self.local_attn_mask = dit.build_local_block_mask_shifted_vec_normal_slide(
                h // 8, w // 8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h // 8
            self.local_attn_mask_w = w // 8
            self.local_range = local_range
        attention_mask = dit.generate_draft_block_mask(
            B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask)

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d',
                      block_n=block_n, block_s=block_s)
        x = dit.WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f * h * w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)

    return forward


def _make_cross_attn_forward(dit):
    import torch as _torch

    def forward(self, x, y, is_stream=False):
        rope_fn = getattr(dit, "FUSED_ROPE_FN", None)
        if rope_fn is not None:
            q = rope_fn(self.q(x), _unwrap(self.norm_q).weight, None,
                        eps=_module_eps(self.norm_q), rope=False)
        else:
            q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v
        x = self.attn(q, k, v)
        return self.o(x)

    return _torch.no_grad()(forward)


def _make_dit_block_forward(dit):
    adaln_pair = lambda: getattr(dit, "FUSED_ADALN", None)

    def forward(self, x, context, t_mod, freqs, f, h, w, local_num=None, topk=None,
                train_img=False, block_id=None, kv_len=None, is_full_block=False,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range=9):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        fa = adaln_pair()
        if fa is not None:
            ln_mod, gate_add = fa
            input_x = ln_mod(x, scale_msa, shift_msa, eps=_module_eps(self.norm1))
        else:
            input_x = dit.modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            input_x, freqs, f, h, w, local_num, topk, train_img, block_id,
            kv_len=kv_len, is_full_block=is_full_block, is_stream=is_stream,
            pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v, local_range=local_range)

        if fa is not None:
            x = gate_add(x, gate_msa, self_attn_output)
        else:
            x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x), context, is_stream=is_stream)
        if fa is not None:
            input_x = ln_mod(x, scale_mlp, shift_mlp, eps=_module_eps(self.norm2))
        else:
            input_x = dit.modulate(self.norm2(x), shift_mlp, scale_mlp)
        if fa is not None:
            x = gate_add(x, gate_mlp, self.ffn(input_x))
        else:
            x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x

    return forward


_EXPECTED_SIGS = {
    "SelfAttention": ("x", "freqs", "f", "h", "w", "local_num", "topk", "train_img",
                      "block_id", "kv_len", "is_full_block", "is_stream",
                      "pre_cache_k", "pre_cache_v", "local_range"),
    "DiTBlock": ("x", "context", "t_mod", "freqs", "f", "h", "w", "local_num", "topk",
                 "train_img", "block_id", "kv_len", "is_full_block", "is_stream",
                 "pre_cache_k", "pre_cache_v", "local_range"),
}


def _check_dit_structure(dit):
    """Cheap guard against upstream drift: the classes we rebind must exist and
    keep the argument names our forwards forward-positionally."""
    import inspect
    for cls, expected in _EXPECTED_SIGS.items():
        c = getattr(dit, cls, None)
        if c is None:
            raise RuntimeError(f"upstream wan_video_dit has no {cls}; this pack "
                               f"targets FlashVSR v1.1 — got {dit}")
        params = tuple(inspect.signature(c.forward).parameters)
        missing = [p for p in expected if p not in params]
        if missing:
            raise RuntimeError(
                f"upstream {cls}.forward lacks args {missing} — upstream signature "
                f"changed since the tested revision (cf910c6); file an issue.")
    if not hasattr(dit, "WindowPartition3D") or not hasattr(dit, "generate_draft_block_mask"):
        raise RuntimeError("upstream wan_video_dit lacks WindowPartition3D / "
                           "generate_draft_block_mask — unsupported revision.")


def _patch_dit(dit, fused_rope, fused_adaln):
    _check_dit_structure(dit)
    dit.FUSED_ROPE_FN = fused_rms_rope if fused_rope else None
    dit.FUSED_ADALN = (fused_ln_modulate, fused_gate_add) if fused_adaln else None
    if getattr(dit, "_FS89_PATCHED", False):
        return  # forwards already rebound; flags above refreshed
    dit.SelfAttention.forward = _make_self_attn_forward(dit)
    dit.CrossAttention.forward = _make_cross_attn_forward(dit)
    dit.DiTBlock.forward = _make_dit_block_forward(dit)
    dit._FS89_PATCHED = True


def enable(pipe, dit_module=None, parts=None, compile_memblocks=None, verbose=True):
    """Apply the full sm_89 operator pack to a FlashVSRTinyPipeline.

    Call once after ``pipe.enable_vram_management(...)`` and before
    ``pipe.init_cross_kv()``. Returns a dict of what was enabled.
    """
    if parts is None:
        parts = [p for p in os.environ.get("FS89_FP8", "ffn,self,cross").split(",") if p]
    if compile_memblocks is None:
        compile_memblocks = _FLAG("FS89_TCDEC_COMPILE", "0")

    applied = {}

    if dit_module is None:
        try:
            import diffsynth.models.wan_video_dit as dit_module
        except ImportError as e:
            raise ImportError(
                "enable() could not import diffsynth.models.wan_video_dit. Run from a "
                "FlashVSR checkout (or pass dit_module=) — see docs/integration.md.") from e

    # 1) TCDecoder channels_last — after vram management; dtype moves reset layout
    if hasattr(pipe, "TCDecoder") and _FLAG("FS89_TCDEC_CL", "1"):
        pipe.TCDecoder.to(memory_format=torch.channels_last)
        applied["tcdec_channels_last"] = True

    # 2) optional MemBlock compile (needs the checkout's utils on sys.path)
    if compile_memblocks and hasattr(pipe, "TCDecoder"):
        try:
            from utils.TCDecoder import MemBlock
        except ImportError:
            raise ImportError(
                "compile_memblocks=True needs the FlashVSR example utils package "
                "(examples/WanVSR) on sys.path — import it before calling enable(), "
                "or leave FS89_TCDEC_COMPILE=0.")
        n = sum(1 for m in pipe.TCDecoder.modules() if isinstance(m, MemBlock))
        for m in pipe.TCDecoder.modules():
            if isinstance(m, MemBlock):
                m.forward = torch.compile(m.forward, mode="max-autotune-no-cudagraphs",
                                          dynamic=False)
        applied["tcdec_compiled_memblocks"] = n

    # 3) FP8 DiT linears (+ fused GELU FFN), idempotent across re-enables
    if parts:
        n = convert_linears_fp8(pipe.denoising_model(), parts=parts)
        applied["fp8_linears"] = n
        if "ffn" in parts and _FLAG("FS89_FP8_FFNFUSED", "1"):
            applied["fp8_ffn_blocks"] = convert_ffn_fp8(pipe.denoising_model())

    # 4) fused bf16 ops via forward rebinding
    _patch_dit(dit_module,
               fused_rope=_FLAG("FS89_FUSED_ROPE", "1"),
               fused_adaln=_FLAG("FS89_FUSED_ADALN", "1"))
    applied["fused_rope"] = _FLAG("FS89_FUSED_ROPE", "1")
    applied["fused_adaln"] = _FLAG("FS89_FUSED_ADALN", "1")

    # 5) LCSA backend — resolved at import of the dit module
    ensure_bsa_available()
    applied["lcsa_backend"] = active_backend()

    if verbose:
        bits = [f"{k}={v}" for k, v in applied.items()]
        print(f"[flashvsr-sm89-ops] enabled: {', '.join(bits)}")
    return applied
