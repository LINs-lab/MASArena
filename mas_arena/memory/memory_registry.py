"""
Benchmark Registry

This module provides a dynamic registry for benchmark evaluators using decorators.
This allows for easy extension by simply decorating a new evaluator class.
"""

import os
from typing import Dict, Any, Type, Optional, List, Callable
from .base import BaseMemory
from .utils import EmbeddingFunc
from .llm import LLMCallable, GPTChat

class MemoryRegistry:
    """A registry for memory classes."""
    _instance = None
    _memories: Dict[str, Dict[str, Any]] = {}
    _instances: Dict[str, BaseMemory] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, name: str) -> Callable:
        """A decorator to register a memory class."""
        def decorator(cls: Type[BaseMemory]) -> Type[BaseMemory]:
            if not issubclass(cls, BaseMemory):
                raise TypeError(f"'{cls.__name__}' must be a subclass of 'BaseMemory'")
            
            if name in self._memories:
                print(f"Warning: Memory '{name}' is already registered and will be overwritten.")
            
            self._memories[name] = {"class": cls}
            return cls
        return decorator

    def get_available_memory_names(self) -> List[str]:
        """Get a list of available memory names."""
        return list(self._memories.keys())

    def get(self, name: str) -> Optional[BaseMemory]:
        """Get an instance of an memory by name."""
        if name not in self._memories:
            raise KeyError(f"Memory '{name}' not found. Available: {', '.join(self.get_available_memory_names())}")
        
        if name not in self._instances:
            memory_info = self._memories[name]
            
            embed_func = EmbeddingFunc(os.environ.get('EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2"))
            llm_model: LLMCallable = GPTChat(model_name=os.environ.get('CHATGPT_MODEL', "gpt-4o-mini"))
            
            kwargs = {
                "namespace": name,
                "llm_model": llm_model,
                "embedding_func": embed_func,
                "persist_dir": os.environ.get('MEMORY_PERSIST_DIR', "mas_arena/memory/persist/melo")
            }
            
            instance = memory_info["class"](**kwargs)
            self._instances[name] = instance
        
        return self._instances[name]


# Global instance
memory_registry = MemoryRegistry()
register_memory = memory_registry.register 