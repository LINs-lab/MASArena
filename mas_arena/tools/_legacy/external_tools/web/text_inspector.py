from typing import Optional
from smolagents import Tool


class TextInspectorTool(Tool):
    """Text inspection tool for analyzing text content."""

    name = "inspect_text_content"
    description = "Inspect and analyze text content from various sources."

    inputs = {
        "text": {
            "type": "string",
            "description": "Text content to inspect and analyze.",
        },
        "analysis_type": {
            "type": "string",
            "description": "[Optional] Type of analysis: 'summary', 'keywords', 'sentiment'. Default is 'summary'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model, text_limit: int = 8000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward(self, text: str, analysis_type: Optional[str] = None) -> str:
        """Analyze text content."""
        analysis = analysis_type or "summary"

        # Truncate text if too long
        if len(text) > self.text_limit:
            text = text[: self.text_limit] + "...[truncated]"

        return f"Text analysis ({analysis}):\n\nInput text length: {len(text)} characters\n\nContent:\n{text}"
