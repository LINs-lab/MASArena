"""
Benchmark Registry

This module provides a dynamic registry for benchmark evaluators using decorators.
This allows for easy extension by simply decorating a new evaluator class.
"""

from typing import Dict, Any, Callable, Type
from .base_evaluator import BaseEvaluator

class BenchmarkRegistry:
    """A registry for benchmark evaluators."""

    def __init__(self):
        """Initializes the benchmark registry."""
        self._benchmarks: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, normalization_keys: Dict[str, str]) -> Callable:
        """
        Returns a decorator that registers an evaluator class for a given mas_arena.

        Args:
            name: The public name of the benchmark (e.g., "math", "humaneval").
            normalization_keys: A dictionary mapping standard data fields to the
                                benchmark's specific field names.

        Returns:
            A decorator for the evaluator class.
        """
        def decorator(cls: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            """The actual decorator that performs the registration."""
            if not issubclass(cls, BaseEvaluator):
                raise TypeError(f"Class {cls.__name__} must be a subclass of BaseEvaluator to be registered.")
            
            if name in self._benchmarks:
                raise ValueError(f"Benchmark '{name}' is already registered.")

            self._benchmarks[name] = {
                "evaluator": cls,
                "normalization_keys": normalization_keys,
            }
            return cls
        return decorator

    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Retrieves the configuration for a given mas_arena.

        Args:
            name: The name of the mas_arena.

        Returns:
            The benchmark configuration dictionary.
        
        Raises:
            KeyError: if the benchmark name is not found.
        """
        if name not in self._benchmarks:
            raise KeyError(f"Benchmark '{name}' not found. Available: {', '.join(self.get_available_benchmark_names())}")
        return self._benchmarks[name]

    def get_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Returns the complete dictionary of all registered benchmarks."""
        return self._benchmarks

    def get_available_benchmark_names(self) -> list[str]:
        """Returns a list of names of all registered benchmarks."""
        return sorted(self._benchmarks.keys())

# Create a global instance of the registry for the application to use.
benchmark_registry = BenchmarkRegistry()

# Create a convenience alias for the registration method to be used as a decorator.
register_benchmark = benchmark_registry.register 