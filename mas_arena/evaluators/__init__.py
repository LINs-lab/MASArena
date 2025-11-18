"""
Evaluators for benchmarking agent systems.

This package dynamically discovers and registers benchmark evaluators.
By importing this package, all evaluator modules are automatically loaded,
which triggers the registration of their respective benchmarks via decorators.
"""

import pkgutil
import importlib

# Import the global registry instance
from .registry import benchmark_registry

# --- Dynamic Discovery and Registration ---
# Iterate over all modules in the current package path
# and import them. This is what triggers the @register_benchmark decorators
# in each evaluator file to run and register themselves.
for _, name, _ in pkgutil.iter_modules(__path__):
    # Ensure we don't try to import the registry module itself again
    # or any other non-evaluator modules.
    if name not in ['registry', 'base_evaluator', 'utils']:
        importlib.import_module(f".{name}", __package__)

# --- Public API ---
# Expose the populated registry and convenient dictionaries for the application.

# The main, dynamically populated dictionary of all benchmark configurations.
# This replaces the old static BENCHMARKS dictionary.
BENCHMARKS = benchmark_registry.get_all_benchmarks()

# For backwards compatibility and convenience, provide a dictionary
# mapping benchmark names to their evaluator classes.
AVAILABLE_EVALUATORS = {name: config["evaluator"] for name, config in BENCHMARKS.items()}

# Define what gets imported when a user does 'from mas_arena.evaluators import *'
__all__ = [
    "benchmark_registry",
    "BENCHMARKS",
    "AVAILABLE_EVALUATORS",
]

