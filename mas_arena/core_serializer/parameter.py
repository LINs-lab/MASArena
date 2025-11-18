from typing import Optional

from .component import SerializableComponent


class Parameter(SerializableComponent):
    name: str
    type: str
    description: str
    required: Optional[bool] = True
