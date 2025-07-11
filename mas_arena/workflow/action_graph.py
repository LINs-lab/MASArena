from ..core.component import SerializableComponent
from pydantic import Field

from ..core.model_configs import LLMConfig


class ActionGraph(SerializableComponent):
    name: str = Field(description="The name of the ActionGraph.")
    description: str = Field(description="The description of the ActionGraph.")
    llm_config: LLMConfig = Field(description="The LLM configuration of the ActionGraph.")
