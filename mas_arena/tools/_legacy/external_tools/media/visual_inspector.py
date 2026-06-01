from typing import Optional

from smolagents import Tool
from smolagents.models import Model


class VisualInspectorTool(Tool):
    name = "inspect_file_as_image"
    description = """
You cannot load files directly: use this tool to process image files and answer related questions.
This tool supports the following image formats: [".jpg", ".jpeg", ".png", ".gif", ".bmp"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as an image. Must be a '.something' file, like '.jpg','.png','.gif'. If it is text, use the text_inspector tool instead! If it is audio, use the audio_inspector tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the image content. Provide as much context as possible. Do not pass this parameter if you just want to get a description of the image.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 1000):
        import os
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.gpt_key = os.getenv("OPENAI_API_KEY")
        self.gpt_url = os.getenv("OPENAI_API_BASE")

    def _validate_file_type(self, file_path: str):
        # NOTE: keep this implementation extremely simple because some tool validators
        # may not understand generator/comprehension local variables (e.g. `ext`).
        supported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
        if not file_path.lower().endswith(supported_extensions):
            raise ValueError(
                "Unsupported file type. Use the appropriate tool for text/audio files."
            )

    def _resize_image(self, image_path: str) -> str:
        import os
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size
        img = img.resize((int(width / 2), int(height / 2)))
        new_image_path = f"resized_{os.path.basename(image_path)}"
        img.save(new_image_path)
        return new_image_path

    def _encode_image(self, image_path: str) -> str:
        import base64
        import mimetypes
        import os
        import uuid

        import requests

        if image_path.startswith("http"):
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            request_kwargs = {
                "headers": {"User-Agent": user_agent},
                "stream": True,
            }

            response = requests.get(image_path, **request_kwargs)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            extension = mimetypes.guess_extension(content_type)
            if extension is None:
                extension = ".download"

            fname = str(uuid.uuid4()) + extension
            download_path = os.path.abspath(os.path.join("downloads", fname))
            os.makedirs(os.path.dirname(download_path), exist_ok=True)

            with open(download_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=512):
                    fh.write(chunk)

            image_path = download_path

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        # Local imports to satisfy tool validators in sandboxed execution
        from typing import Optional
        import mimetypes
        import requests

        self._validate_file_type(file_path)

        if not question:
            question = "Please write a detailed caption for this image."
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            base64_image = self._encode_image(file_path)
            payload = {
                "model": "gpt-4o-2024-11-20",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 1000,
                "top_p": 0.1,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gpt_key}",
            }

            response = requests.post(
                f"{self.gpt_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            description = response.json()["choices"][0]["message"]["content"]
        except Exception as gpt_error:
            return f"Visual processing failed: {str(gpt_error)}"

        if not question.startswith("Please write a detailed caption"):
            return description
        return f"You did not provide a particular question, so here is a detailed description of the image: {description}"
