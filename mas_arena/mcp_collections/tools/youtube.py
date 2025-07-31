"""
YouTube MCP Service

This module provides MCP server functionality for YouTube operations including:
- Downloading videos from YouTube URLs
- Extracting transcripts from YouTube videos

It handles various scenarios with proper validation, error handling,
and progress tracking while providing LLM-friendly formatted results.
"""

import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from youtube_transcript_api import FetchedTranscript, YouTubeTranscriptApi

from mas_arena.mcp_collections.base import ActionArguments, ActionResponse, ActionCollection

# Default driver path for Chrome WebDriver
_DEFAULT_DRIVER_PATH = os.environ.get(
    "CHROME_DRIVER_PATH", str(Path("~/Downloads/chromedriver-mac-x64/chromedriver").expanduser())
)


class YoutubeDownloadResults(BaseModel):
    """Download result model with file information"""

    file_path: str
    file_name: str
    file_size: int
    content_type: str | None = None
    success: bool
    error: str | None = None


class TranscriptResult(BaseModel):
    """Transcript result model with transcript information"""

    video_id: str
    transcript: FetchedTranscript
    success: bool
    error: str | None = None


class YouTubeMetadata(BaseModel):
    """Metadata for YouTube operation results"""

    operation: str
    url: str | None = None
    video_id: str | None = None
    file_path: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    content_type: str | None = None
    language_code: str | None = None
    translate_to_language: str | None = None
    execution_time: float | None = None
    error_type: str | None = None


class YouTubeActionCollection(ActionCollection):
    """MCP service for YouTube operations.

    Provides YouTube capabilities including:
    - Video downloading with Selenium automation
    - Transcript extraction and translation
    - LLM-friendly result formatting
    - Error handling and logging
    """
    tool_name = "youtube"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Initialize supported file extensions
        self.supported_extensions = {".mp4", ".webm", ".mkv"}


    def _format_transcript_output(self, result: TranscriptResult, format_type: str = "markdown") -> str:
        """Format transcript results for LLM consumption.

        Args:
            result: Transcript extraction result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if result is None or not result.success:
            return f"Failed to extract transcript: {result.error}"

        if format_type == "json":
            return result.model_dump()
        elif format_type == "text":
            output = [f"Transcript for video ID: {result.video_id}\n"]

            # Access snippets from FetchedTranscript
            for entry in result.transcript.snippets:
                start_time = entry["start"]
                text = entry["text"]

                minutes, seconds = divmod(int(start_time), 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                output.append(f"[{timestamp}] {text}")

            return "\n".join(output)
        else:  # markdown (default)
            output = [f"# Transcript for YouTube Video: {result.video_id}\n"]
            output.append("| Timestamp | Text |")
            output.append("| --- | --- |")

            # Access snippets from FetchedTranscript
            for entry in result.transcript.snippets:
                start_time = entry["start"]
                text: str = entry["text"]

                minutes, seconds = divmod(int(start_time), 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                # Escape pipe characters in markdown table
                safe_text = text.replace("|", "\\|")
                output.append(f"| {timestamp} | {safe_text} |")

            return "\n".join(output)

    def _format_download_output(self, result: YoutubeDownloadResults, format_type: str = "markdown") -> str:
        """Format download results for LLM consumption.

        Args:
            result: Download result
            format_type: Output format ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not result.success:
            return f"Failed to download video: {result.error}"

        if format_type == "json":
            return result.model_dump()
        elif format_type == "text":
            output_parts = [
                "Download completed successfully",
                f"File: {result.file_name}",
                f"Path: {result.file_path}",
                f"Size: {result.file_size} bytes",
            ]
            if result.content_type:
                output_parts.append(f"Content Type: {result.content_type}")

            return "\n".join(output_parts)
        else:  # markdown (default)
            output_parts = [
                "# YouTube Download Results ✅",
                "",
                "## File Information",
                f"**Filename:** `{result.file_name}`",
                f"**Path:** `{result.file_path}`",
                f"**Size:** {result.file_size} bytes",
            ]
            if result.content_type:
                output_parts.append(f"**Content Type:** {result.content_type}")

            return "\n".join(output_parts)

    def _get_youtube_content(self, url: str, output_dir: str, timeout: int) -> None:
        """Use Selenium to download YouTube content via cobalt.tools

        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded content
            timeout: Maximum time to wait for download in seconds
        """
        driver = None
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            # Set download file default path
            prefs = {
                "download.default_directory": output_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            }
            options.add_experimental_option("prefs", prefs)
            # Create WebDriver object and launch Chrome browser
            service = Service(executable_path=_DEFAULT_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
            
            # Add to active drivers list for cleanup
            if not hasattr(self.__class__, '_active_drivers'):
                self.__class__._active_drivers = []
            self.__class__._active_drivers.append(driver)

            print(f"Opening cobalt.tools to download from {url}")
            # Open target webpage
            driver.get("https://cobalt.tools/")
            # Wait for page to load
            time.sleep(5)
            # Find input field and enter YouTube link
            input_field = driver.find_element(By.ID, "link-area")
            input_field.send_keys(url)
            time.sleep(5)
            # Find download button and click
            download_button = driver.find_element(By.ID, "download-button")
            download_button.click()
            time.sleep(5)

            try:
                # Handle bot detection popup
                driver.find_element(
                    By.CLASS_NAME,
                    "button.elevated.popup-button.undefined.svelte-nnawom.active",
                ).click()
            except Exception as e:
                print(f"Bot detection handling: {str(e)}")

            # try:
            #     t = 0
            #     while t < timeout:
            #         if (
            #             "downloading" not in driver.find_element(By.CLASS_NAME, "status-text.svelte-dmosdd").text
            #             and "starting" not in driver.find_element(By.CLASS_NAME, "status-text.svelte-dmosdd").text
            #         ):
            #             driver.find_element(By.CLASS_NAME, "button.action-button.svelte-dmosdd").click()
            #             break
            #         t += 3
            #         time.sleep(3)
            # except Exception as e:
            #     print(f"Bot detection handling: {str(e)}")

            # Wait for download to complete
            cnt = 0
            while len(os.listdir(output_dir)) == 0 or os.listdir(output_dir)[0].split(".")[-1] == "crdownload":
                time.sleep(3)
                cnt += 3
                if cnt >= timeout:
                    print(f"Download timeout after {timeout} seconds")
                    break

            print("Download process completed")

        except Exception as e:
            print(f"Error during YouTube content download: {str(e)}")
            raise
        finally:
            # Close browser
            if driver:
                try:
                    driver.quit()
                    # Remove from active drivers list
                    if hasattr(self.__class__, '_active_drivers') and driver in self.__class__._active_drivers:
                        self.__class__._active_drivers.remove(driver)
                except Exception as e:
                    print(f"Error closing browser: {e}")

    def _find_existing_video(self, search_dir: str, video_id: str) -> str | None:
        """Recursively search for an existing video file with the given ID.

        Args:
            search_dir: Directory to search in
            video_id: YouTube video ID to look for

        Returns:
            Path to existing file if found, None otherwise
        """
        if not video_id:
            return None

        search_path = Path(search_dir)
        if not search_path.exists():
            return None

        for item in search_path.iterdir():
            if item.is_file() and video_id in item.name:
                return str(item)
            elif item.is_dir():
                found = self._find_existing_video(str(item), video_id)
                if found:
                    return found

        return None

    # Track active browser instances for proper cleanup
    _active_drivers = []
    
    async def mcp_download_youtube_video(
        self,
        url: str = Field(description="The URL of YouTube video to download."),
        timeout: int = Field(180, description="Download timeout in seconds (default: 180)."),
        output_format: str = Field(
            "markdown", description="Output format: 'markdown', 'json', or 'text' (default: markdown)."
        ),
    ) -> ActionResponse:
        """Download a YouTube video from URL and save it to the local filesystem."""
        start_time = time.time()

        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                raise ValueError("Invalid URL format. URL must start with http:// or https://")

            if not ("youtube.com" in url or "youtu.be" in url):
                raise ValueError("URL must be a valid YouTube URL")

            # Create output directory if it doesn't exist
            output_path = self.workspace / "youtube_downloads"
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename based on timestamp
            filename = f"youtube_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_path = output_path / filename
            file_path.mkdir(parents=True, exist_ok=True)
            print(f"Output path: {file_path}")

            # Extract video ID for existing file check
            video_id = url.split("?v=")[-1].split("&")[0] if "?v=" in url else ""
            if "youtu.be/" in url and not video_id:
                video_id = url.split("youtu.be/")[-1].split("?")[0]

            # Check if video already exists
            base_path = self.workspace
            existing_file = self._find_existing_video(str(base_path), video_id)

            if existing_file:
                existing_path = Path(existing_file)
                result = YoutubeDownloadResults(
                    file_path=str(existing_path),
                    file_name=existing_path.name,
                    file_size=existing_path.stat().st_size,
                    content_type="mp4",
                    success=True,
                    error=None,
                )
                print(f"Found {video_id} already downloaded in: {existing_file}")

                # Format output for LLM
                message = self._format_download_output(result, output_format)
                execution_time = time.time() - start_time

                # Create metadata
                metadata = YouTubeMetadata(
                    operation="download",
                    url=url,
                    video_id=video_id,
                    file_path=str(existing_path),
                    file_name=existing_path.name,
                    file_size=existing_path.stat().st_size,
                    content_type="mp4",
                    execution_time=execution_time,
                ).model_dump()

                return ActionResponse(success=True, message=message, metadata=metadata)

            # Download the video
            print(f"Downloading video from {url} to {file_path}")
            self._get_youtube_content(url, str(file_path), timeout)

            # Check if download was successful
            downloaded_files = list(file_path.iterdir())
            if not downloaded_files:
                raise FileNotFoundError("No files were downloaded")

            download_file = downloaded_files[0]
            file_size = download_file.stat().st_size

            print(f"File downloaded successfully to {download_file}")

            # Create result
            result = YoutubeDownloadResults(
                file_path=str(download_file),
                file_name=download_file.name,
                file_size=file_size,
                content_type="mp4",
                success=True,
                error=None,
            )

            # Format output for LLM
            message = self._format_download_output(result, output_format)
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="download",
                url=url,
                video_id=video_id,
                file_path=str(download_file),
                file_name=download_file.name,
                file_size=file_size,
                content_type="mp4",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            error_msg = str(e)
            print(f"Download error: {traceback.format_exc()}")

            # Format error for LLM
            message = f"Failed to download YouTube video: {error_msg}"
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="download",
                url=url,
                error_type="download_failure",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=False, message=message, metadata=metadata)

    async def mcp_extract_youtube_transcript(
        self,
        video_id: str = Field(description="The YouTube video ID or URL to extract transcript from."),
        language_code: str = Field("en", description="Language code for the transcript (default: en)."),
        translate_to_language: str | None = Field(
            None, description="Translate transcript to this language code if provided."
        ),
        output_format: str = Field(
            "markdown", description="Output format: 'markdown', 'json', or 'text' (default: markdown)."
        ),
    ) -> ActionResponse:
        """Extract transcript from a YouTube video given its video ID or URL."""
        start_time = time.time()

        try:
            # Clean video_id if full URL was provided
            if "youtube.com" in video_id or "youtu.be" in video_id:
                if "?v=" in video_id:
                    video_id = video_id.split("?v=")[-1].split("&")[0]
                elif "youtu.be/" in video_id:
                    video_id = video_id.split("youtu.be/")[-1].split("?")[0]

            print(f"Extracting transcript for video ID: {video_id}")

            # Get transcript in specified language
            if translate_to_language:
                # Get transcript and translate it
                y_api = YouTubeTranscriptApi()
                transcript_list = y_api.list(video_id)
                transcript = None

                try:
                    # Try to get transcript in specified language
                    transcript = transcript_list.find_transcript([language_code])
                except Exception:
                    # If specified language not found, get any available transcript
                    transcript = transcript_list.find_generated_transcript(["en"])

                # Translate to target language
                transcript_data = transcript.translate(translate_to_language).fetch()

            else:
                try:
                    # Get transcript without translation
                    transcript_data: FetchedTranscript = (
                        YouTubeTranscriptApi()
                        .list(video_id)
                        .find_transcript((language_code,))
                        .fetch(preserve_formatting=False)
                    )
                except Exception:
                    transcript_data = None

            result = TranscriptResult(video_id=video_id, transcript=transcript_data, success=True, error=None)

            print(f"Successfully extracted transcript for video ID: {video_id}")

            # Format output for LLM
            message = self._format_transcript_output(result, output_format)
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="transcript",
                video_id=video_id,
                language_code=language_code,
                translate_to_language=translate_to_language,
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=True, message=message, metadata=metadata)

        except Exception as e:
            error_msg = str(e)
            print(f"Transcript extraction error: {traceback.format_exc()}")

            # Format error for LLM
            message = f"Failed to extract transcript: {error_msg}"
            execution_time = time.time() - start_time

            # Create metadata
            metadata = YouTubeMetadata(
                operation="transcript",
                video_id=video_id,
                language_code=language_code,
                translate_to_language=translate_to_language,
                error_type="transcript_failure",
                execution_time=execution_time,
            ).model_dump()

            return ActionResponse(success=False, message=message, metadata=metadata)


# Default arguments for testing
# Add a cleanup method to ensure all browser instances are closed
def cleanup_all_drivers():
    """Ensure all browser instances are properly closed."""
    if hasattr(YouTubeActionCollection, '_active_drivers'):
        for driver in YouTubeActionCollection._active_drivers.copy():
            try:
                if driver:
                    driver.quit()
            except Exception as e:
                print(f"Error closing browser during cleanup: {e}")
        YouTubeActionCollection._active_drivers.clear()

# Register cleanup function to run at exit
import atexit
atexit.register(cleanup_all_drivers)

if __name__ == "__main__":
    import sys
    import json
    
    # Determine if running in MCP tool mode (called without arguments)
    is_mcp_mode = len(sys.argv) == 1

    # Redirect print to stderr if in MCP mode
    if is_mcp_mode:
        original_print = print
        print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
    
    load_dotenv()

    # Default arguments for testing
    arguments = ActionArguments(
        name="youtube_service",
        transport="stdio",
        workspace=os.getenv("MASARENA_WORKSPACE", "~"),
    )

    # Initialize and run the YouTube service
    try:
        youtube_service = YouTubeActionCollection(arguments)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                # Use function_name instead of name for better semantic consistency
                function_name = input_data.get("function_name", input_data.get("name", ""))
                arguments = input_data.get("arguments", {})
                
                if function_name == "mcp_download_youtube_video":
                    result = youtube_service.mcp_download_youtube_video(
                        url=arguments.get("url", ""),
                        timeout=arguments.get("timeout", 180),
                        output_format=arguments.get("output_format", "markdown")
                    )
                elif function_name == "mcp_extract_youtube_transcript":
                    result = youtube_service.mcp_extract_youtube_transcript(
                        video_id=arguments.get("video_id", ""),
                        language_code=arguments.get("language_code", "en"),
                        translate_to_language=arguments.get("translate_to_language", None),
                        output_format=arguments.get("output_format", "markdown")
                    )
                else:
                    result = ActionResponse(
                        success=False,
                        message=f"Unknown function: {function_name}",
                        metadata={"error_type": "unknown_function"}
                    )
                
                # Convert result to synchronous response if it's awaitable
                import asyncio
                if asyncio.iscoroutine(result):
                    result = asyncio.run(result)
                
                # Write result to stdout as JSON
                sys.stdout.write(json.dumps(result.model_dump()) + "\n")
                sys.stdout.flush()
                sys.exit(0)
            except json.JSONDecodeError as e:
                error = {"success": False, "message": f"Invalid JSON input: {str(e)}"}
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()
                sys.exit(1)
            except Exception as e:
                error = {"success": False, "message": f"Error: {str(e)}"}
                sys.stderr.write(f"Exception: {traceback.format_exc()}\n")
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()
                sys.exit(1)
        else:
            # Normal MCP server mode
            youtube_service.run()
    except Exception as e:
        sys.stderr.write(f"An error occurred: {e}: {traceback.format_exc()}\n")
        sys.exit(1)
