from pydantic import Field
from mas_arena.agent_flow.base_config import BaseConfig


class LLMConfig(BaseConfig):
    llm_type: str
    model: str
    output_response: bool = Field(default=False, description="Whether to output the response.")
