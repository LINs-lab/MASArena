# coding: utf-8
from typing import List

from langchain.tools import StructuredTool

from mas_arena.tools.base import ToolFactory

ANDROID = "android"

class Android:
    def tap(self, x: int, y: int) -> str:
        """Simulate a tap on the screen at the given coordinates."""
        return f"Simulated tap at ({x}, {y})"

    def get_screen(self) -> str:
        """Simulate getting the current screen content."""
        return "Simulated screen content as a string."

    def type_text(self, text: str) -> str:
        """Simulate typing text."""
        return f"Simulated typing text: '{text}'"

@ToolFactory.register(name=ANDROID, desc="A tool for interacting with an Android device.")
class AndroidTool:
    def __init__(self):
        self.android = Android()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.android.tap,
                name="tap_screen",
                description="Simulate a tap on the screen at the given coordinates.",
            ),
            StructuredTool.from_function(
                func=self.android.get_screen,
                name="get_screen_content",
                description="Simulate getting the current screen content.",
            ),
            StructuredTool.from_function(
                func=self.android.type_text,
                name="type_text",
                description="Simulate typing text.",
            ),
        ] 