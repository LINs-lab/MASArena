# Compatibility shim: internal legacy tools expect mas_arena.tools.base.ToolFactory
from mas_arena.tools._legacy.base import ToolFactory  # noqa: F401

__all__ = ["ToolFactory"]
