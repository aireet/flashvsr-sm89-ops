"""ComfyUI loader entry — the node pack itself lives in ./comfyui.

Clone this whole repository into ComfyUI/custom_nodes; nothing needs to be
pip-installed (this file is not part of the PyPI package).
"""
from .comfyui import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
