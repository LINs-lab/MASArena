import time
from pathlib import Path
from typing import Optional

import chardet
from smolagents import Tool


class TextExtractorTool(Tool):
    name = "extract_text_content"
    description = """
Extract content from text documents with encoding detection and analysis.
This tool supports various text file formats: TXT, MD, CSV, JSON, XML, YAML, source code files, logs, and more.
Provides comprehensive text analysis with statistics and LLM-friendly formatted output.
"""

    inputs = {
        "file_path": {
            "description": "The path to the text document file to extract content from. Supports formats: .txt, .md, .csv, .json, .xml, .yaml, .py, .js, .html, .css, .log, and many more text-based files.",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the text content. Provide as much context as possible. If not provided, returns the full content with analysis.",
            "type": "string",
            "nullable": True,
        },
        "max_length": {
            "description": "[Optional]: Maximum length of content to include in output. If not specified, no truncation is applied.",
            "type": "number",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, text_limit: int = 10000):
        super().__init__()
        self.text_limit = text_limit
        self.supported_extensions = {
            ".txt",
            ".text",
            ".log",
            ".md",
            ".markdown",
            ".rst",
            ".rtf",
            ".csv",
            ".tsv",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".sql",
            ".py",
            ".js",
            ".html",
            ".htm",
            ".css",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".sh",
            ".bat",
            ".ps1",
            ".r",
            ".m",
            ".swift",
            ".kt",
            ".scala",
            ".pl",
            ".lua",
            ".vim",
            ".tex",
            ".bib",
        }

    def _validate_file_path(self, file_path: str) -> Path:
        """Validate and resolve file path."""
        from pathlib import Path
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check if it's a supported extension or appears to be text
        if path.suffix.lower() not in self.supported_extensions:
            # Try to detect if it's a text file by reading a small sample
            try:
                with open(path, "rb") as f:
                    sample = f.read(1024)
                if not self._is_likely_text(sample):
                    raise ValueError(
                        f"Unsupported file type: {path.suffix}. Use the appropriate tool for binary files."
                    )
            except Exception:
                raise ValueError(f"Cannot read file or unsupported format: {path}")

        return path

    def _is_likely_text(self, data: bytes) -> bool:
        """Check if binary data is likely to be text."""
        if not data:
            return True

        # Check for null bytes (common in binary files)
        if b"\x00" in data:
            return False

        # Try to decode as text
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            pass

        # Check if most bytes are printable ASCII
        printable_count = 0
        for i in range(len(data)):
            char_code = data[i]
            # 检查是否为可打印 ASCII 或常见的制表/换行符
            if (32 <= char_code <= 126) or (char_code in [9, 10, 13]):
                printable_count += 1
        
        return (printable_count / len(data)) > 0.7

    def _detect_encoding(self, file_path: Path) -> dict:
        """Detect file encoding and characteristics."""
        import chardet
        encoding_info = {
            "detected_encoding": "utf-8",
            "confidence": 0.0,
            "line_endings": "Unknown",
            "is_binary": False,
        }

        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()

            if not raw_data:
                return encoding_info

            # Use chardet for encoding detection
            detection_result = chardet.detect(raw_data)
            encoding_info["detected_encoding"] = detection_result.get(
                "encoding", "utf-8"
            )
            encoding_info["confidence"] = detection_result.get("confidence", 0.0)

            # Detect line endings
            if b"\r\n" in raw_data:
                encoding_info["line_endings"] = "CRLF (Windows)"
            elif b"\n" in raw_data:
                encoding_info["line_endings"] = "LF (Unix/Linux/Mac)"
            elif b"\r" in raw_data:
                encoding_info["line_endings"] = "CR (Classic Mac)"

            # Check if file appears to be binary
            encoding_info["is_binary"] = not self._is_likely_text(raw_data[:1024])

        except Exception:
            # Use default values
            pass

        return encoding_info

    def _detect_content_type(self, content: str, file_path: Path) -> str:
        """Detect the type of content based on file extension and content patterns."""
        extension = file_path.suffix.lower()

        extension_map = {
            ".py": "Python source code",
            ".js": "JavaScript source code",
            ".html": "HTML document",
            ".htm": "HTML document",
            ".css": "CSS stylesheet",
            ".json": "JSON data",
            ".xml": "XML document",
            ".yaml": "YAML configuration",
            ".yml": "YAML configuration",
            ".md": "Markdown document",
            ".markdown": "Markdown document",
            ".rst": "reStructuredText document",
            ".csv": "CSV data",
            ".tsv": "TSV data",
            ".sql": "SQL script",
            ".log": "Log file",
            ".ini": "Configuration file",
            ".cfg": "Configuration file",
            ".conf": "Configuration file",
        }

        if extension in extension_map:
            return extension_map[extension]

        # Content-based detection
        content_lower = content.lower().strip()

        if content_lower.startswith("<?xml"):
            return "XML document"
        elif content_lower.startswith("{") and content_lower.endswith("}"):
            return "JSON-like data"
        elif content_lower.startswith("[") and content_lower.endswith("]"):
            return "JSON array or configuration"
        elif "#!/" in content[:50]:
            return "Script file"
        elif content.count(",") > content.count("\n") * 2:
            return "CSV-like data"
        else:
            return "Plain text"

    def _extract_text_content(self, file_path: Path) -> dict:
        """Extract content from text files with comprehensive analysis."""
        start_time = time.time()

        # Detect encoding
        encoding_info = self._detect_encoding(file_path)
        target_encoding = encoding_info["detected_encoding"]

        try:
            # Read file content
            with open(file_path, "r", encoding=target_encoding, errors="replace") as f:
                content = f.read()

            # Analyze content
            lines = content.splitlines()

            # Calculate statistics
            char_count = len(content)
            line_count = len(lines)
            word_count = len(content.split()) if content.strip() else 0

            # Line length analysis
            line_lengths = [len(line) for line in lines]
            max_line_length = max(line_lengths) if line_lengths else 0
            min_line_length = min(line_lengths) if line_lengths else 0
            avg_line_length = (
                sum(line_lengths) / len(line_lengths) if line_lengths else 0
            )

            # Count empty lines
            empty_lines = sum(1 for line in lines if not line.strip())

            # Detect content type
            content_type = self._detect_content_type(content, file_path)

            processing_time = time.time() - start_time

            return {
                "content": content,
                "encoding_info": encoding_info,
                "statistics": {
                    "character_count": char_count,
                    "line_count": line_count,
                    "word_count": word_count,
                    "empty_lines": empty_lines,
                    "max_line_length": max_line_length,
                    "min_line_length": min_line_length,
                    "avg_line_length": round(avg_line_length, 2),
                },
                "content_type": content_type,
                "processing_time": processing_time,
                "used_encoding": target_encoding,
            }

        except UnicodeDecodeError:
            # Try with fallback encodings
            fallback_encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

            for fallback_encoding in fallback_encodings:
                try:
                    with open(
                        file_path, "r", encoding=fallback_encoding, errors="replace"
                    ) as f:
                        content = f.read()

                    # Update encoding info
                    encoding_info["detected_encoding"] = fallback_encoding
                    encoding_info["confidence"] = 0.5  # Lower confidence for fallback

                    # Quick analysis for fallback
                    lines = content.splitlines()
                    char_count = len(content)
                    line_count = len(lines)
                    word_count = len(content.split()) if content.strip() else 0

                    return {
                        "content": content,
                        "encoding_info": encoding_info,
                        "statistics": {
                            "character_count": char_count,
                            "line_count": line_count,
                            "word_count": word_count,
                            "empty_lines": 0,
                            "max_line_length": 0,
                            "min_line_length": 0,
                            "avg_line_length": 0,
                        },
                        "content_type": self._detect_content_type(content, file_path),
                        "processing_time": time.time() - start_time,
                        "used_encoding": fallback_encoding,
                        "encoding_fallback": True,
                    }
                except:
                    continue

            raise ValueError("Unable to decode file with any supported encoding")

    def _format_content_for_analysis(
        self, extraction_result: dict, max_length: Optional[int] = None
    ) -> str:
        """Format extracted text content for LLM analysis."""
        content = extraction_result["content"]
        stats = extraction_result["statistics"]
        content_type = extraction_result["content_type"]
        encoding_info = extraction_result["encoding_info"]

        # Truncate content if needed
        original_length = len(content)
        if max_length and len(content) > max_length:
            content = content[:max_length]
            truncated = True
        else:
            truncated = False

        # Format the output
        formatted_parts = []
        formatted_parts.append("# Text Document Analysis\n")
        formatted_parts.append(f"**File Type:** {content_type}\n")
        formatted_parts.append(
            f"**Encoding:** {extraction_result['used_encoding']} (confidence: {encoding_info['confidence']:.2f})\n"
        )
        formatted_parts.append(f"**Line Endings:** {encoding_info['line_endings']}\n")

        if extraction_result.get("encoding_fallback"):
            formatted_parts.append(
                "⚠️ **Warning:** Used fallback encoding - content may not be fully accurate\n"
            )

        if encoding_info["is_binary"]:
            formatted_parts.append(
                "⚠️ **Warning:** File appears to contain binary data\n"
            )

        formatted_parts.append("\n**Statistics:**\n")
        formatted_parts.append(f"- Characters: {stats['character_count']:,}\n")
        formatted_parts.append(f"- Lines: {stats['line_count']:,}\n")
        formatted_parts.append(f"- Words: {stats['word_count']:,}\n")
        formatted_parts.append(f"- Empty lines: {stats['empty_lines']:,}\n")

        if stats["avg_line_length"] > 0:
            formatted_parts.append(
                f"- Average line length: {stats['avg_line_length']} characters\n"
            )
            formatted_parts.append(
                f"- Longest line: {stats['max_line_length']} characters\n"
            )

        formatted_parts.append(
            f"- Processing time: {extraction_result['processing_time']:.3f} seconds\n\n"
        )

        if truncated:
            formatted_parts.append(
                f"**Content** (showing first {max_length:,} of {original_length:,} characters):\n\n"
            )
        else:
            formatted_parts.append("**Content:**\n\n")

        # Add content in code block for better formatting
        if content_type in ["JSON data", "XML document", "YAML configuration"]:
            lang = content_type.split()[0].lower()
            formatted_parts.append(f"```{lang}\n{content}\n```")
        elif "source code" in content_type:
            lang = content_type.split()[0].lower()
            formatted_parts.append(f"```{lang}\n{content}\n```")
        else:
            formatted_parts.append(f"```\n{content}\n```")

        if truncated:
            formatted_parts.append(
                f"\n\n*[Content truncated - showing first {max_length:,} characters of {original_length:,} total]*"
            )

        return "".join(formatted_parts)

    def forward(
        self,
        file_path: str,
        question: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> str:
        import time
        from pathlib import Path
        """Extract and analyze text content from files."""
        try:
            # Validate file path
            path = self._validate_file_path(file_path)

            # Extract content
            extraction_result = self._extract_text_content(path)

            # Use provided max_length or default text_limit
            effective_max_length = max_length or self.text_limit

            # Format content for analysis
            formatted_content = self._format_content_for_analysis(
                extraction_result, effective_max_length
            )

            # If no specific question, return the formatted content
            if not question:
                return formatted_content

            # If there's a question, provide the content with context for the question
            content_for_question = extraction_result["content"]
            if (
                effective_max_length
                and len(content_for_question) > effective_max_length
            ):
                content_for_question = content_for_question[:effective_max_length]

            analysis_context = f"""Text Document Analysis Context:
File: {path.name}
Type: {extraction_result['content_type']}
Encoding: {extraction_result['used_encoding']}
Statistics: {extraction_result['statistics']['character_count']:,} characters, {extraction_result['statistics']['line_count']:,} lines, {extraction_result['statistics']['word_count']:,} words

Content:
{content_for_question}

Question: {question}

Please answer the question based on the text content above. Provide:
1. Direct answer to the question
2. Relevant details from the text
3. Any important context or patterns you notice"""

            return analysis_context

        except FileNotFoundError as e:
            return f"Error: File not found - {str(e)}"
        except ValueError as e:
            return f"Error: Invalid file or unsupported format - {str(e)}"
        except Exception as e:
            return f"Error: Text extraction failed - {str(e)}"
