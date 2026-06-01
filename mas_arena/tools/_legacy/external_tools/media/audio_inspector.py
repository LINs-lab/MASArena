from typing import Optional, Dict, Any
import os
from datetime import timedelta

from smolagents import Tool
from smolagents.models import MessageRole, Model

import openai
from mutagen._file import File as MutagenFile


class AudioInspectorTool(Tool):
    name = "inspect_file_as_audio"
    description = """
You cannot load files directly: use this tool to process audio files and answer related questions.
This tool supports the following audio formats: [".mp3", ".m4a", ".wav"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as audio. Must be a '.something' file, like '.mp3','.m4a','.wav'. If it is an text, use the text_inspector tool instead! If it is an image, use the visual_inspector tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the audio content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.api_key = os.getenv("AUDIO_LLM_API_KEY")
        self.base_url = os.getenv("AUDIO_LLM_BASE_URL")

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported audio format"""
        # NOTE: keep this implementation extremely simple because some tool validators
        # may not understand generator/comprehension local variables (e.g. `ext`).
        supported_extensions = (".mp3", ".m4a", ".wav")
        if not file_path.endswith(supported_extensions):
            raise ValueError(
                "Unsupported file type. Use the appropriate tool for text/image files."
            )

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file using mutagen"""
        try:
            audio_file = MutagenFile(file_path)
            if audio_file is None:
                return {"error": "Unsupported audio format or corrupted file"}

            metadata = {}

            # Basic file information
            metadata["file_size"] = f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"
            metadata["format"] = audio_file.mime[0] if audio_file.mime else "Unknown"

            # Audio properties
            if hasattr(audio_file, "info"):
                info = audio_file.info
                if hasattr(info, "length"):
                    duration = timedelta(seconds=int(info.length))
                    metadata["duration"] = str(duration)
                if hasattr(info, "bitrate"):
                    metadata["bitrate"] = f"{info.bitrate} bps"
                if hasattr(info, "sample_rate"):
                    metadata["sample_rate"] = f"{info.sample_rate} Hz"
                if hasattr(info, "channels"):
                    metadata["channels"] = info.channels

            # Common tags
            tag_mapping = {
                "TIT2": "title",  # ID3v2.4
                "TPE1": "artist",  # ID3v2.4
                "TALB": "album",  # ID3v2.4
                "TDRC": "date",  # ID3v2.4
                "TCON": "genre",  # ID3v2.4
                "TRCK": "track",  # ID3v2.4
                # Common tag names for other formats
                "TITLE": "title",
                "ARTIST": "artist",
                "ALBUM": "album",
                "DATE": "date",
                "GENRE": "genre",
                "TRACKNUMBER": "track",
                # MP4 tags
                "©nam": "title",
                "©ART": "artist",
                "©alb": "album",
                "©day": "date",
                "©gen": "genre",
                "trkn": "track",
            }

            # Extract tags
            if audio_file.tags:
                for key, value in audio_file.tags.items():
                    if key in tag_mapping:
                        clean_key = tag_mapping[key]
                        if isinstance(value, list) and len(value) > 0:
                            metadata[clean_key] = str(value[0])
                        else:
                            metadata[clean_key] = str(value)

            return metadata

        except Exception as e:
            return {"error": f"Metadata extraction failed: {str(e)}"}

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", file=audio_file
                )
            return transcription.text
        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {str(e)}") from e

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        # 确认文件类型
        self._validate_file_type(file_path)

        # 提取音频元信息
        metadata = self.extract_metadata(file_path)

        # 格式化元信息
        metadata_str = "Audio Metadata:\n"
        for key, value in metadata.items():
            metadata_str += f"  {key.replace('_', ' ').title()}: {value}\n"
        metadata_str += "\n"

        # 获取音频文件描述
        try:
            transcript = self.transcribe_audio(file_path)
        except Exception as e:
            return f"Audio processing error: {str(e)}\n\n{metadata_str}"

        if not question:
            return f"{metadata_str}Audio transcription:\n{transcript[:self.text_limit]}"

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": f"{metadata_str}Here is the an audio transcription:\n{transcript[:self.text_limit]}\n"
                        "Answer the following question based on the audio content using the format:1. Brief answer\n2. Detailed analysis\n3. Relevant context\n\n",
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
