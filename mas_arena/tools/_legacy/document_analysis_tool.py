# coding: utf-8
import base64
import json
from typing import List, Dict, Any
import logging

from langchain.tools import StructuredTool

from mas_arena.tools.base import ToolFactory
from mas_arena.tools.document import Document

logger = logging.getLogger(__name__)

def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_url(url):
    try:
        import requests
    except ImportError:
        raise ImportError("requests library not found. Please install requests: pip install requests")
    response = requests.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')

DOCUMENT_ANALYSIS = "document_analysis"

class DocumentAnalysis:
    def read_document(self, file_path: str) -> str:
        """
        Read and parse content from a document file.
        Supports various formats including PDF, DOCX, PPTX, images, audio, video, etc.
        Returns a JSON string containing the extracted content, keyframes (for videos), and any errors.
        """
        try:
            doc = Document()
            content, keyframes, error = doc.document_analysis(file_path)

            result: Dict[str, Any] = {"content": content}
            if keyframes:
                result["keyframes"] = keyframes
            if error:
                result["error"] = error
            
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}, ensure_ascii=False)

@ToolFactory.register(name=DOCUMENT_ANALYSIS, desc="A tool for reading and analyzing various document types.")
class DocumentAnalysisTool:
    def __init__(self):
        self.document_analysis = DocumentAnalysis()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.document_analysis.read_document,
                name="read_document",
                description="Read and parse content from a document file. Supports various formats including PDF, DOCX, PPTX, images, audio, video, etc.",
            ),
        ] 