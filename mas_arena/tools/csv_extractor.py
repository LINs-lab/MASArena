import time
import json
from pathlib import Path
from typing import Optional

import chardet
import pandas as pd
from smolagents import Tool
from smolagents.models import MessageRole, Model


class CSVExtractorTool(Tool):
    name = "inspect_file_as_csv"
    description = """
Extract and analyze content from CSV, TSV, and delimited text files.
This tool supports the following formats: [".csv", ".tsv", ".txt"]. 
For other file types, use the appropriate inspection tool.
Provides automatic encoding and delimiter detection, statistical analysis, and multiple output formats."""

    inputs = {
        "file_path": {
            "description": "The path to the CSV file you want to analyze. Must be a CSV, TSV, or delimited text file (.csv, .tsv, .txt). If it is an image, use the visual_inspector tool instead! If it is audio, use the audio_inspector tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the CSV data content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
        "output_format": {
            "description": "[Optional]: Output format for the data. Options: 'markdown', 'json', 'html', 'text'. Default is 'markdown'.",
            "type": "string",
            "nullable": True,
        },
        "max_rows": {
            "description": "[Optional]: Maximum number of rows to read from the CSV file. Default is None (all rows).",
            "type": "integer",
            "nullable": True,
        },
        "include_statistics": {
            "description": "[Optional]: Whether to include statistical summary in output. Default is True.",
            "type": "boolean",
            "nullable": True,
        },
        "encoding": {
            "description": "[Optional]: File encoding (e.g., 'utf-8', 'latin-1'). Auto-detected if not provided.",
            "type": "string",
            "nullable": True,
        },
        "delimiter": {
            "description": "[Optional]: CSV delimiter character (e.g., ',', ';', '\t'). Auto-detected if not provided.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model =None, text_limit: int = 50000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.supported_extensions = {".csv", ".tsv", ".txt"}

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported CSV format"""
        is_valid = False
        for ext in self.supported_extensions:
            if file_path.lower().endswith(ext):
                is_valid = True
                break
        
        if not is_valid:
            raise ValueError(
                f"Unsupported file type. Supported: {list(self.supported_extensions)}"
            )

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        import chardet
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result.get("encoding", "utf-8")
                confidence = result.get("confidence", 0)
                return encoding if encoding and confidence > 0.7 else "utf-8"
        except Exception:
            return "utf-8"

    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter by analyzing the first few lines."""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                sample = f.read(1024)  # Read first 1KB

            # Common delimiters to test
            delimiters = [",", ";", "\t", "|", ":"]
            delimiter_counts = {}

            for delimiter in delimiters:
                count = sample.count(delimiter)
                if count > 0:
                    delimiter_counts[delimiter] = count

            if delimiter_counts:
                detected_delimiter = max(
                    delimiter_counts.keys(), key=lambda x: delimiter_counts[x]
                )
                return detected_delimiter
            else:
                return ","
        except Exception:
            return ","

    def _extract_csv_content(
        self,
        file_path: Path,
        max_rows: Optional[int] = None,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> dict:
        import pandas as pd
        import time
        """Extract content from CSV file using pandas."""
        start_time = time.time()

        # Auto-detect encoding and delimiter if not provided
        if encoding is None:
            encoding = self._detect_encoding(file_path)
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path, encoding)

        try:
            # Read CSV with pandas
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                nrows=max_rows,
                low_memory=False,
            )

            # Count total rows efficiently
            total_rows = (
                sum(1 for _ in open(file_path, "r", encoding=encoding)) - 1
            )  # Subtract header

            processing_time = time.time() - start_time

            return {
                "dataframe": df,
                "total_rows": total_rows,
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "encoding": encoding,
                "delimiter": delimiter,
                "processing_time": processing_time,
                "data_types": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")

    def _format_content_for_llm(
        self,
        df: pd.DataFrame,
        output_format: str = "markdown",
        include_stats: bool = True,
    ) -> str:
        import json
        """Format extracted CSV content to be LLM-friendly."""
        if output_format.lower() == "markdown":
            # Convert to markdown table
            content = df.to_markdown(index=False, tablefmt="github")

            if include_stats:
                # Add statistical summary
                stats_content = "\n\n## Data Summary\n\n"
                stats_content += f"- **Rows**: {len(df)}\n"
                stats_content += f"- **Columns**: {len(df.columns)}\n"
                stats_content += f"- **Column Names**: {', '.join(df.columns)}\n\n"

                # Add data types info
                stats_content += "### Column Data Types\n\n"
                for col, dtype in df.dtypes.items():
                    stats_content += f"- **{col}**: {dtype}\n"

                # Add basic statistics for numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    stats_content += "\n### Numeric Column Statistics\n\n"
                    stats_df = df[numeric_cols].describe()
                    stats_content += stats_df.to_markdown(tablefmt="github")

                content += stats_content

        elif output_format.lower() == "json":
            # Convert to JSON with metadata
            data_dict = {
                "data": df.to_dict(orient="records"),
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                },
            }
            if include_stats:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    data_dict["statistics"] = df[numeric_cols].describe().to_dict()

            content = json.dumps(data_dict, indent=2, default=str)

        elif output_format.lower() == "html":
            # Convert to HTML table
            content = df.to_html(index=False, classes="table table-striped")

        else:
            # Plain text format
            content = df.to_string(index=False)

        return content

    def forward(
        self,
        file_path: str,
        question: Optional[str] = None,
        output_format: Optional[str] = "markdown",
        max_rows: Optional[int] = None,
        include_statistics: Optional[bool] = True,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> str:
        from pathlib import Path  
        from smolagents.models import MessageRole
        """Main method to process CSV files and return formatted content."""

        # Validate file type
        self._validate_file_type(file_path)

        # Convert to Path object
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            return f"Error: File not found: {file_path}"

        try:
            # Extract CSV content
            extraction_result = self._extract_csv_content(
                file_path_obj, max_rows=max_rows, encoding=encoding, delimiter=delimiter
            )

            df = extraction_result["dataframe"]

            # Format content for LLM consumption
            formatted_content = self._format_content_for_llm(
                df,
                output_format or "markdown",
                include_stats=(
                    include_statistics if include_statistics is not None else True
                ),
            )

            # Add metadata information
            metadata_info = (
                f"\n\n**File Processing Info:**\n"
                f"- File: {file_path_obj.name}\n"
                f"- Total rows: {extraction_result['total_rows']}\n"
                f"- Total columns: {extraction_result['total_columns']}\n"
                f"- Encoding: {extraction_result['encoding']}\n"
                f"- Delimiter: '{extraction_result['delimiter']}'\n"
                f"- Processing time: {extraction_result['processing_time']:.2f}s\n"
                f"- Memory usage: {extraction_result['memory_usage']/1024/1024:.2f} MB"
            )

            # Limit content length
            full_content = formatted_content + metadata_info
            if len(full_content) > self.text_limit:
                full_content = (
                    full_content[: self.text_limit]
                    + "\n\n[Content truncated due to length limit]"
                )

            # If no question provided, return the formatted content directly
            if not question:
                return full_content

            # Use the model to answer questions about the CSV data
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is the CSV file content and analysis:\n\n{full_content}\n\n"
                            "Answer the following question based on the CSV data using the format:\n"
                            "1. Brief answer\n2. Detailed analysis\n3. Relevant data context\n\n",
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please answer the question: {question}",
                        }
                    ],
                },
            ]

            content = self.model(messages).content
            if not isinstance(content, str):
                content = str(content)
            return content

        except Exception as e:
            return f"CSV processing error: {str(e)}"
