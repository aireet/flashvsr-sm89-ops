"""FlashVSR-sm89 ComfyUI node pack: VIDEO in -> VIDEO out, local, on your GPU.

Imports the operator pack from the repository checkout beside it, so a plain
`git clone` of this repo into ComfyUI/custom_nodes is the whole install.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
