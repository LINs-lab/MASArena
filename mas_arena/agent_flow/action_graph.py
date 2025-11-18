from mas_arena.agent_flow.model_configs import LLMConfig
from mas_arena.core_serializer.component import SerializableComponent
from pydantic import Field


class ActionGraph(SerializableComponent):
    name: str = Field(description="The name of the ActionGraph.")
    description: str = Field(description="The description of the ActionGraph.")
    llm_config: LLMConfig = Field(description="The LLM configuration of the ActionGraph.")
