from typing import Optional

from smolagents import Tool
from smolagents.models import MessageRole, Model

import os
from markitdown import MarkItDown


class MarkdownConverterTool(Tool):
    name = "convert_to_markdown"
    description = """
You cannot load files directly: use this tool to convert various document types to markdown format.
This tool supports the following formats: PDF, PowerPoint (.pptx), Word (.docx), Excel (.xlsx), Images (with OCR), Audio (with transcription), HTML, CSV, JSON, XML, ZIP files, Youtube URLs, and EPubs.
For pure text files or already markdown files, use the text_inspector tool instead!"""

    inputs = {
        "file_path": {
            "description": "The path to the file or URL you want to convert to markdown. Supports: PDF, PowerPoint (.pptx), Word (.docx), Excel (.xlsx), Images, Audio, HTML, CSV, JSON, XML, ZIP, Youtube URLs, EPubs.",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the converted content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the markdown content of the file.",
            "type": "string",
            "nullable": True,
        },
        "enable_plugins": {
            "description": "[Optional]: Whether to enable MarkItDown plugins for enhanced conversion. Default is False.",
            "type": "boolean",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 50000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.supported_extensions = [
            ".pdf",
            ".pptx",
            ".docx",
            ".xlsx",
            ".xls",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".mp3",
            ".wav",
            ".m4a",
            ".mp4",
            ".avi",
            ".mov",
            ".html",
            ".htm",
            ".csv",
            ".json",
            ".xml",
            ".zip",
            ".epub",
        ]

    def _validate_file_or_url(self, file_path: str):
        import os
        """Validate if the file type is supported or if it's a URL"""
        # Check if it's a YouTube URL
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return True

        if file_path.startswith(("http://", "https://")):
            return True

        is_supported_ext = False
        for ext in self.supported_extensions:
            if file_path.lower().endswith(ext):
                is_supported_ext = True
                break
        
        if is_supported_ext:
            return True

        if os.path.exists(file_path):
            return True

        return False

    def convert_to_markdown(self, file_path: str, enable_plugins: bool = False) -> str:
        """Convert file to markdown using MarkItDown"""
        from markitdown import MarkItDown
        try:
            md = MarkItDown(enable_plugins=enable_plugins)
            result = md.convert(file_path)
            return result.text_content
        except Exception as e:
            raise RuntimeError(f"Document conversion failed: {str(e)}") from e

    def forward(
        self,
        file_path: str,
        question: Optional[str] = None,
        enable_plugins: Optional[bool] = None,
    ) -> str:
        from smolagents.models import MessageRole
        # Set default for enable_plugins
        if enable_plugins is None:
            enable_plugins = False

        # Validate file/URL
        if not self._validate_file_or_url(file_path):
            return f"Unsupported file type or file not found: {file_path}. Please use the appropriate tool for text files or check if the file exists."

        # Convert document to markdown
        try:
            markdown_content = self.convert_to_markdown(file_path, enable_plugins)
        except Exception as e:
            return f"Document conversion error: {str(e)}"

        # If no question is provided, return the markdown content directly
        if not question:
            return f"Markdown content:\n{markdown_content[:self.text_limit]}"

        # If a question is provided, use the model to answer based on the markdown content
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the markdown content converted from the document:\n{markdown_content[:self.text_limit]}\n"
                        "Answer the following question based on the document content using the format:\n1. Brief answer\n2. Detailed analysis\n3. Relevant context\n\n",
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {"type": "text", "text": f"Please answer the question: {question}"}
                ],
            },
        ]

        content = self.model(messages).content
        if not isinstance(content, str):
            content = str(content)
        return content
