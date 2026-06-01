"""
LangGraph Multi-Agent System Benchmark Framework.

This package provides tools for benchmarking and evaluating the performance of
multi-agent systems built with LangGraph, focusing on system-level metrics
and performance characteristics.
"""

__version__ = "0.1.0"

import importlib
import sys

from mas_arena.metrics import (
    MetricsRegistry,
)

_legacy_tools = importlib.import_module("mas_arena.tools._legacy")
sys.modules.setdefault("mas_arena.tools_old", _legacy_tools)
sys.modules.setdefault("mas_arena.legacy_tools", _legacy_tools)

__all__ = [
    "MetricsRegistry",
]
