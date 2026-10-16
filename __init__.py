"""Fish Audio Drama 3 (fishaudio/drama-3, multi-speaker text to speech) as native ComfyUI nodes."""
from .nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
