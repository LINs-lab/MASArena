import threading
from openai import OpenAI
import pandas as pd
from dataclasses import dataclass
import os
from mas_arena.agents import AgentSystem
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any


@dataclass
class Cost:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float


class CostManager:

    def __init__(self):

        self.total_input_tokens = {}
        self.total_output_tokens = {}
        self.total_tokens = {}

        self.total_input_cost = {}
        self.total_output_cost = {}
        self.total_cost = {}

        self._lock = threading.Lock()

    def get_total_cost(self):

        total_cost = 0.0
        for model in self.total_cost.keys():
            total_cost += self.total_cost[model]
        return total_cost


cost_manager = CostManager()

def get_agent_by_name(agent_name: str) -> AgentSystem:
    from mas_arena.agents import AgentSystemRegistry
    config = {}
    agent_system = AgentSystemRegistry.get(agent_name, config)
    if agent_system is None:
        raise ValueError(f"Agent system '{agent_name}' not found in registry.")
    return agent_system

def call_model(query, model_name, key=None, url=None):
    if len(query) > 300000:
        query = query[:300000]

    api_key = key if key else os.environ.get("CHATGPT_API_KEY")
    base_url = url if url else os.environ.get("OPENAI_API_BASE")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided. Please set the OPENAI_API_BASE environment variable or pass it as an argument.")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        extra_body={},
        model=model_name,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
            ]
            }
        ]
    )
    return completion.choices[0].message.content

class RetryWrapper:
    """A wrapper to add retry logic to an LLM model's API calls."""
    def __init__(self, model, max_retries=3):
        self._model = model
        self.max_retries = max_retries
        # 保存原始模型ID以防被修改
        if hasattr(self._model, 'model_id'):
            self._original_model_id = self._model.model_id

    def _should_retry(self, exception: BaseException) -> bool:
        """Return True if we should retry on this exception."""
        if isinstance(exception, httpx.HTTPStatusError):
            # Retry on 5xx server errors, including 524
            return 500 <= exception.response.status_code < 600 or exception.response.status_code == 400
        if isinstance(exception, httpx.TimeoutException):
            return True
        return False

    def _preserve_model_id(self):
        """确保在重试前恢复原始的模型ID"""
        if hasattr(self, '_original_model_id') and hasattr(self._model, 'model_id'):
            # 在每次调用前恢复原始模型ID
            if self._model.model_id != self._original_model_id:
                print(f"修复模型ID: 从 {self._model.model_id} 恢复为 {self._original_model_id}")
                self._model.model_id = self._original_model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=_should_retry
    )
    def __call__(self, *args, **kwargs) -> Any:
        """Delegate the call to the wrapped model with retry logic."""
        # 在调用模型前确保模型ID正确
        self._preserve_model_id()
        
        # 如果在kwargs中有model参数，确保它也是正确的
        if 'model' in kwargs and hasattr(self, '_original_model_id'):
            if kwargs['model'] != self._original_model_id and self._original_model_id in str(kwargs['model']):
                print(f"修复kwargs中的模型ID: 从 {kwargs['model']} 恢复为 {self._original_model_id}")
                kwargs['model'] = self._original_model_id
                
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model."""
        attr = getattr(self._model, name)
        
        # 如果返回的是一个方法，确保在调用前恢复模型ID
        if callable(attr):
            def wrapper(*args, **kwargs):
                self._preserve_model_id()
                return attr(*args, **kwargs)
            return wrapper
        return attr