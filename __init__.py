"""ComfyUI entry point — clone this whole repository into ComfyUI/custom_nodes.

The node loads the sibling flashvsr_sm89_ops package straight from this
checkout, so no pip install is needed inside ComfyUI (pip users install the
package normally; this file is not part of the PyPI package).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
