# Compatibility shim: internal legacy tools expect mas_arena.tools.document.Document
from mas_arena.tools._legacy.document import Document  # noqa: F401

__all__ = ["Document"]
