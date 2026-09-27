"""Defuse import-time hard dependencies upstream FlashVSR pulls in but never
lists in its own requirements.txt.

- ``diffsynth/models/downloader.py`` does ``from modelscope import
  snapshot_download`` at module scope, so ``from diffsynth import ...`` crashes
  on any fresh environment without modelscope — even though loading the release
  weights from a local directory never touches the downloader. Importing
  ``flashvsr_sm89_ops`` calls :func:`ensure_modelscope_available`, which installs
  a stub module **only when modelscope is absent**; the stub raises a clear
  error if the download path is ever actually used.

- diffsynth predates transformers v5, which stopped re-exporting
  ``PretrainedConfig``/``PreTrainedModel`` from ``transformers.modeling_utils``.
  :func:`ensure_transformers_compat` aliases the names back (a no-op on the
  transformers 4.x versions upstream pins).
"""
import sys
import types

STUB_VERSION = "flashvsr-sm89-ops/modelscope-stub"


def _snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "modelscope is not installed. diffsynth only needs it for its "
        "model-downloader presets; FlashVSR release weights load fine from a "
        "local directory without it. To use the downloader, run "
        "`pip install modelscope`.")


def ensure_modelscope_available():
    """Stub ``modelscope`` unless the real package is installed."""
    if "modelscope" in sys.modules:
        return "modelscope"
    try:
        import modelscope  # noqa: F401
        return "modelscope"
    except ImportError:
        pass
    stub = types.ModuleType("modelscope")
    stub.snapshot_download = _snapshot_download
    stub.__version__ = STUB_VERSION
    sys.modules["modelscope"] = stub
    return "stub"


def ensure_transformers_compat():
    """Restore the transformers-4 names diffsynth imports from
    ``modeling_utils`` when running on transformers >= 5."""
    try:
        import transformers
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        return
    for name in ("PretrainedConfig", "PreTrainedModel"):
        if not hasattr(modeling_utils, name) and hasattr(transformers, name):
            setattr(modeling_utils, name, getattr(transformers, name))
