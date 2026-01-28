# Compatibility shim: tools_old imports expect mas_arena.tools.base.ToolFactory
from mas_arena.tools_old.base import ToolFactory  # noqa: F401

__all__ = ["ToolFactory"]
