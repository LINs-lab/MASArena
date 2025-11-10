from dataclasses import dataclass, field
from typing import Any, Optional
import json
import threading
from contextlib import contextmanager

@dataclass
class StorageNameSpace:
    """
    StorageNameSpace represents a namespace for storage-related tasks,
    such as indexing and querying.

    Attributes:
        namespace (str): The identifier for this storage namespace.
    """
    namespace: str

    def _index_done(self):
        pass

    def _query_done(self):
        pass
@dataclass
class MASMessage:
    final_answer: str
    ground_truth: str
    task_question: str
    task_search_keywords: str
    task_trajectory: Optional[str] = '\n\n>'
    label: Optional[bool] = None
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    
    # def add_message_to_current_state(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
    #     return self.chain_of_states.add_message(agent_message, upstream_agent_ids)
    
    # def move_state(self, action: str, observation: str, **args) -> None:
    #     self.task_trajectory += f'{action}\n{observation}\n>'
    #     self.chain_of_states.move_state(action, observation, **args)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None)
    
    @staticmethod
    def to_dict(mas_message: "MASMessage") -> dict[str, str]:
        return {
            "final_answer": mas_message.final_answer,
            "ground_truth": mas_message.ground_truth,
            "task_question": mas_message.task_question,
            "task_search_keywords": mas_message.task_search_keywords,
            "task_trajectory": mas_message.task_trajectory,
            "label": mas_message.label,
            "extra_fields": json.dumps(mas_message.extra_fields),
        }
    
    @staticmethod
    def from_dict(message_dict: dict) -> "MASMessage":
        return MASMessage(
            final_answer=message_dict.get("final_answer"),
            ground_truth=message_dict.get("ground_truth"),
            task_question=message_dict.get("task_question"),
            task_search_keywords=message_dict.get("task_search_keywords"),
            task_trajectory=message_dict.get("task_trajectory"),
            label=message_dict.get("label"),
            extra_fields=json.loads(message_dict.get("extra_fields", "{}")),
        )


class RWLock:
    """
    A simple reader-writer lock implementation that prefers writers.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    @contextmanager
    def read_lock(self):
        """Context manager for acquiring a read lock."""
        with self._cond:
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1
        
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self):
        """Context manager for acquiring a write lock."""
        with self._cond:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True
        
        try:
            yield
        finally:
            with self._cond:
                self._writer_active = False
                self._cond.notify_all()