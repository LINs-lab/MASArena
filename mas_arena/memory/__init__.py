

import pkgutil
import importlib

from .memory_registry import memory_registry


for _, name, _ in pkgutil.iter_modules(__path__):
    # Ensure we don't try to import the registry module itself again
    # or any other non-evaluator modules.
    if name not in ['memory_registry', 'base', 'utils']:
        importlib.import_module(f".{name}", __package__)


MEMORIES = memory_registry.get_available_memory_names()


AVAILABLE_MEMORIES = {name for name in MEMORIES}

# Define what gets imported when a user does 'from mas_arena.evaluators import *'
__all__ = [
    "memory_registry",
    "MEMORIES",
    "AVAILABLE_MEMORIES",
]

