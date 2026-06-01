from typing import Optional, Dict, Any
import os
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from smolagents import Tool
from smolagents.models import MessageRole, Model


class DownloadTool(Tool):
    name = "download_file"
    description = """
Download files from HTTP/HTTPS URLs with comprehensive options and controls.
This tool supports downloading files from web URLs and saves them to local paths.
Features include configurable timeout, overwrite protection, and detailed progress reporting."""

    inputs = {
        "url": {
            "description": "HTTP/HTTPS URL of the file to download. Must be a valid URL starting with http:// or https://",
            "type": "string",
        },
        "output_file_path": {
            "description": "Local path where the file should be saved. Can be absolute or relative to workspace. Parent directories will be created automatically.",
            "type": "string",
        },
        "overwrite": {
            "description": "[Optional] Whether to overwrite existing files. Default is False.",
            "type": "boolean",
            "nullable": True,
        },
        "timeout": {
            "description": "[Optional] Download timeout in seconds. Default is 60 seconds.",
            "type": "integer",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, workspace: Optional[str] = None):
        super().__init__()
        self.model = model

        # Initialize workspace
        if workspace:
            self.workspace = Path(os.path.expanduser(workspace))
        else:
            self.workspace = Path(os.path.expanduser("~"))

        # Configuration
        self.default_timeout = 60
        self.max_file_size = 1024 * 1024 * 1024  # 1GB limit
        self.supported_schemes = {"http", "https"}

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate URL format and scheme."""
        try:
            parsed = urlparse(url)

            if not parsed.scheme:
                return False, "URL must include a scheme (http:// or https://)"

            if parsed.scheme.lower() not in self.supported_schemes:
                return (
                    False,
                    f"Unsupported URL scheme: {parsed.scheme}. Supported: {', '.join(self.supported_schemes)}",
                )

            if not parsed.netloc:
                return False, "URL must include a valid domain"

            return True, ""

        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"

    def _resolve_output_path(self, output_path: str) -> Path:
        """Resolve and validate output file path."""
        path = Path(output_path).expanduser()

        if not path.is_absolute():
            path = self.workspace / path

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        return path.resolve()

    def _download_file(
        self, url: str, output_path: Path, timeout: int
    ) -> Dict[str, Any]:
        """Download file with comprehensive error handling."""
        start_time = datetime.now()

        try:
            print(f"📥 Starting download: {url}")

            with requests.get(
                url, stream=True, timeout=timeout, headers=self.headers
            ) as response:
                response.raise_for_status()

                # Check for incorrect content type when downloading videos
                content_type = response.headers.get("content-type", "").lower()
                video_extensions = {".mp4", ".webm", ".mkv", ".flv", ".avi", ".mov", ".wmv"}
                if "text/html" in content_type and output_path.suffix.lower() in video_extensions:
                    raise ValueError(
                        f"Attempting to download a video but the URL points to an HTML page (Content-Type: {content_type}). "
                        "This tool cannot download videos from streaming sites like YouTube. Please provide a direct link to the video file."
                    )

                # Check content length if available
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(
                        f"File too large: {content_length} bytes (max: {self.max_file_size})"
                    )

                # Download file
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)

                file_size = output_path.stat().st_size
                duration = datetime.now() - start_time

                print(f"✅ Download completed: {file_size:,} bytes")

                return {
                    "success": True,
                    "file_size": file_size,
                    "duration": str(duration),
                    "content_type": response.headers.get("content-type", "Unknown"),
                    "status_code": response.status_code,
                }

        except requests.exceptions.Timeout:
            duration = datetime.now() - start_time
            error_msg = f"Download timed out after {timeout} seconds"
            print(f"⏰ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "duration": str(duration),
            }

        except requests.exceptions.RequestException as e:
            duration = datetime.now() - start_time
            error_msg = f"Request failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "duration": str(duration),
            }

        except Exception as e:
            duration = datetime.now() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "duration": str(duration),
            }

    def _format_result(
        self, url: str, output_path: Path, result: Dict[str, Any]
    ) -> str:
        """Format download results for LLM consumption."""
        timestamp = datetime.now().isoformat()

        if result["success"]:
            file_size_mb = result["file_size"] / (1024 * 1024)

            output_parts = [
                "# File Download ✅",
                f"**URL:** `{url}`",
                f"**File Path:** `{output_path}`",
                f"**Status:** SUCCESS",
                f"**File Size:** {result['file_size']:,} bytes ({file_size_mb:.2f} MB)",
                f"**Content Type:** {result.get('content_type', 'Unknown')}",
                f"**Duration:** {result['duration']}",
                f"**Timestamp:** {timestamp}",
            ]

            if result.get("status_code"):
                output_parts.append(f"**HTTP Status:** {result['status_code']}")

            return "\n".join(output_parts)
        else:
            output_parts = [
                "# File Download ❌",
                f"**URL:** `{url}`",
                f"**File Path:** `{output_path}`",
                f"**Status:** FAILED",
                f"**Duration:** {result['duration']}",
                f"**Timestamp:** {timestamp}",
                "\n## Error Details",
                f"```\n{result['error']}\n```",
            ]

            return "\n".join(output_parts)

    def forward(
        self,
        url: str,
        output_file_path: str,
        overwrite: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Download a file from URL to local path."""

        # Set defaults for optional parameters
        if overwrite is None:
            overwrite = False
        if timeout is None:
            timeout = self.default_timeout

        try:
            # Validate URL
            url_valid, url_error = self._validate_url(url)
            if not url_valid:
                return f"❌ Invalid URL: {url_error}"

            # Resolve output path
            output_path = self._resolve_output_path(output_file_path)

            # Check if file exists and overwrite setting
            if output_path.exists() and not overwrite:
                existing_size = output_path.stat().st_size
                return f"❌ File already exists at {output_path} ({existing_size:,} bytes) and overwrite is disabled"

            # Perform download
            result = self._download_file(url, output_path, timeout)

            # Format and return result
            return self._format_result(url, output_path, result)

        except Exception as e:
            return f"❌ Failed to download file: {str(e)}"

if __name__ == "__main__":
    download_tool = DownloadTool(model=Model(name="gpt-4o-mini"), workspace=".")
    result = download_tool.forward(
        url="https://www.youtube.com/watch?v=1htKBjuUWec",
        output_file_path="video.mp4",
        overwrite=True,
        timeout=60
    )
    print(result)