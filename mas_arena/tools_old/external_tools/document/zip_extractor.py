import os
import zipfile
from typing import Optional, List

from smolagents import Tool
from smolagents.models import MessageRole, Model


class ZipExtractorTool(Tool):
    name = "extract_zip_file"
    description = """
Extract ZIP files and return a listing of extracted files.
This tool supports ZIP file format and extracts all files to a specified directory.
Use this tool when you need to extract and analyze the contents of ZIP archives."""

    inputs = {
        "file_path": {
            "description": "The path to the ZIP file you want to extract. Must be a '.zip' file.",
            "type": "string",
        },
        "extract_dir": {
            "description": "[Optional]: The directory where files will be extracted. Defaults to 'extracted_files'",
            "type": "string",
            "nullable": True,
        },
        "question": {
            "description": "[Optional]: Your question about the extracted files. Provide as much context as possible. Do not pass this parameter if you just want to directly return the list of extracted files.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int = 8000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported ZIP format"""
        if not file_path.lower().endswith(".zip"):
            raise ValueError(
                "Unsupported file type. This tool only supports ZIP files."
            )

        if not zipfile.is_zipfile(file_path):
            raise ValueError("The file is not a valid ZIP archive.")

    def extract_zip_files(
        self, file_path: str, extract_dir: str = "extracted_files"
    ) -> List[str]:
        """Extract ZIP files and return list of extracted file paths"""
        # Create the extraction directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)

        extracted_files = []

        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_dir)

                # Get list of all files (skip directories)
                for file_path_in_zip in zip_ref.namelist():
                    if not file_path_in_zip.endswith("/"):
                        extracted_file_path = os.path.join(
                            extract_dir, file_path_in_zip
                        )
                        extracted_files.append(extracted_file_path)

            # Sort files for consistent output
            extracted_files.sort()
            return extracted_files

        except Exception as e:
            raise RuntimeError(f"ZIP extraction failed: {str(e)}") from e

    def forward(
        self,
        file_path: str,
        extract_dir: Optional[str] = None,
        question: Optional[str] = None,
    ) -> str:
        # Validate file type
        self._validate_file_type(file_path)

        # Set default extraction directory
        if extract_dir is None:
            extract_dir = "extracted_files"

        # Extract ZIP files
        try:
            extracted_files = self.extract_zip_files(file_path, extract_dir)
        except Exception as e:
            return f"ZIP extraction error: {str(e)}"

        # Build the file listing content
        md_content = f"Extracted {len(extracted_files)} files from ZIP archive:\n"
        for file in extracted_files:
            md_content += f"* {file}\n"

        # If no question is provided, return the file listing directly
        if not question:
            return md_content.strip()

        # If a question is provided, use the model to analyze the extracted files
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the list of files extracted from a ZIP archive:\n{md_content[:self.text_limit]}\n"
                        "Answer the following question based on the extracted file information using the format:\n"
                        "1. Brief answer\n2. Detailed analysis\n3. Relevant context\n\n",
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
