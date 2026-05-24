import threading
from openai import OpenAI
import pandas as pd
from dataclasses import dataclass
import os
from mas_arena.agents import AgentSystem
import httpx
from tenacity import retry, stop_after_attempt, retry_if_exception
from typing import Any

from mas_arena.utils.chatgpt_keys import get_next_chatgpt_api_key

DEFAULT_LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "20"))
RATE_LIMIT_RETRY_WAIT_SECONDS = int(os.environ.get("LLM_RETRY_WAIT_SECONDS", "60"))
TRANSIENT_RETRY_WAIT_SECONDS = RATE_LIMIT_RETRY_WAIT_SECONDS


def _get_status_code(exception: BaseException) -> int | None:
    code = getattr(exception, "status_code", None)
    if isinstance(exception, httpx.HTTPStatusError):
        code = exception.response.status_code
    return code if isinstance(code, int) else None


def _retry_wait_seconds(retry_state: Any) -> float:
    return RATE_LIMIT_RETRY_WAIT_SECONDS


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

    api_key = key if key else get_next_chatgpt_api_key()
    base_url = url if url else os.environ.get("OPENAI_API_BASE")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided. Please set CHATGPT_API_KEY or pass it as an argument.")
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
    def __init__(self, model, max_retries=None):
        self._model = model
        self.max_retries = max_retries if max_retries is not None else DEFAULT_LLM_MAX_RETRIES
        # 保存原始模型ID以防被修改
        if hasattr(self._model, 'model_id'):
            self._original_model_id = self._model.model_id
    
    
    @staticmethod
    def _should_retry(exception: BaseException) -> bool:
        """Return True if we should retry on this exception."""
        # OpenAI SDK exceptions expose status_code directly, while httpx stores it
        # on the response. Treat 408/429/5xx as transient in both cases.
        code = _get_status_code(exception)
        if isinstance(code, int):
            return 500 <= code < 600 or code in (408, 429)
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

    def __call__(self, *args, **kwargs) -> Any:
        """Delegate the call to the wrapped model with retry logic."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=_retry_wait_seconds,
            retry=retry_if_exception(self._should_retry),
        )
        def _invoke() -> Any:
            # 在调用模型前确保模型ID正确
            self._preserve_model_id()

            # 如果在kwargs中有model参数，确保它也是正确的
            call_kwargs = kwargs
            if 'model' in call_kwargs and hasattr(self, '_original_model_id'):
                if (
                    call_kwargs['model'] != self._original_model_id
                    and self._original_model_id in str(call_kwargs['model'])
                ):
                    print(
                        f"修复kwargs中的模型ID: 从 {call_kwargs['model']} "
                        f"恢复为 {self._original_model_id}"
                    )
                    call_kwargs = {**call_kwargs, 'model': self._original_model_id}

            return self._model(*args, **call_kwargs)

        return _invoke()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped model."""
        attr = getattr(self._model, name)

        if callable(attr):
            import asyncio as _asyncio
            from tenacity import retry as _retry, stop_after_attempt as _stop, retry_if_exception as _rif

            if _asyncio.iscoroutinefunction(attr):
                async def async_wrapper(*args, **kwargs):
                    @_retry(
                        stop=_stop(self.max_retries),
                        wait=_retry_wait_seconds,
                        retry=_rif(self._should_retry),
                    )
                    async def _invoke():
                        self._preserve_model_id()
                        return await attr(*args, **kwargs)
                    return await _invoke()
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    @_retry(
                        stop=_stop(self.max_retries),
                        wait=_retry_wait_seconds,
                        retry=_rif(self._should_retry),
                    )
                    def _invoke():
                        self._preserve_model_id()
                        return attr(*args, **kwargs)
                    return _invoke()
                return sync_wrapper

        return attr