import os

from typing import (
    Protocol,
    Literal,
    Optional,
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from dotenv import load_dotenv
load_dotenv()

URL = os.environ["OPENAI_API_BASE"]
KEY = os.environ["CHATGPT_API_KEY"]
print('# api url: ', URL)
print('# api key: ', KEY)


def _get_env_int(var_name: str, default_value: int) -> int:
    """Read int env var safely with fallback to default."""
    raw_value = os.getenv(var_name)
    if raw_value is None or raw_value == "":
        return default_value
    try:
        return int(raw_value)
    except Exception:
        return default_value


def _get_env_float(var_name: str, default_value: float) -> float:
    """Read float env var safely with fallback to default."""
    raw_value = os.getenv(var_name)
    if raw_value is None or raw_value == "":
        return default_value
    try:
        return float(raw_value)
    except Exception:
        return default_value


# Centralized, typed defaults derived from environment variables
DEFAULT_TEMPERATURE: float = _get_env_float("TEMPERATURE", 0.1)
DEFAULT_MAX_TOKENS: int = _get_env_int("MAX_TOKENS", 2048)
DEFAULT_NUM_COMPS: int = _get_env_int("NUM_COMPS", 1)


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):
    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )
        # Persist typed defaults to avoid re-reading env and to ensure numeric types
        self.default_temperature: float = DEFAULT_TEMPERATURE
        self.default_max_tokens: int = DEFAULT_MAX_TOKENS
        self.default_num_comps: int = DEFAULT_NUM_COMPS

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        import time
        global prompt_tokens, completion_tokens
        
        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

        # Resolve effective numeric parameters with proper typing
        temp: float = float(temperature) if temperature is not None else self.default_temperature
        mx: int = int(max_tokens) if max_tokens is not None else self.default_max_tokens
        num: int = int(num_comps) if num_comps is not None else self.default_num_comps

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_tokens=mx,
                    temperature=temp,
                    n=num,
                    stop=stop_strs
                )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    print("Error: LLM returned None")
                    continue
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break 

        return "" 


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000
