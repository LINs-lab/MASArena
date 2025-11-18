import os
from dataclasses import dataclass
from abc import ABC

from .common import (
    MASMessage,
    StorageNameSpace
)
from mas_arena.memory.llm import LLMCallable
from mas_arena.memory.utils import EmbeddingFunc


@dataclass
class BaseMemory(StorageNameSpace, ABC):
    """
    Abstract base class for managing multi-agent system (MAS) memory within a namespace.
    This class handles the lifecycle of task contexts, including creation, updating, saving,
    and retrieval of memory states associated with multi-agent tasks.


    Post-initialization:
        Creates a directory for persisting memory data based on global configuration and namespace.
    """
    llm_model: LLMCallable
    embedding_func: EmbeddingFunc
    persist_dir: str
    
    def __post_init__(self):
        if self.persist_dir is None:
            self.persist_dir = os.environ.get('PERSIST_DIR', "mas_arena/memory/persist/melo")
        os.makedirs(self.persist_dir, exist_ok=True)

    async def add_memory(self, mas_message: MASMessage):
        pass

    async def retrieve_memory(self, **kargs) -> tuple[list, list, list]:
        return [], [], []

    async def update_memory(self, query: str, **kargs) -> None:
        pass

    def backward(self, reward, **kwargs) -> None:
        pass
