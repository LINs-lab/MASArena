# Compatibility shim: tools_old imports expect mas_arena.tools.document.Document
from mas_arena.tools_old.document import Document  # noqa: F401

__all__ = ["Document"]
