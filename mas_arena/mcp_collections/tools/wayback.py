"""
Wayback Machine MCP Server

This module provides MCP server functionality for interacting with the Wayback Machine.
It supports listing archived versions, fetching archived content, and saving pages to the archive.

Key features:
- List available archived versions of URLs with date filtering
- Fetch content from specific archived page versions
- Save current pages to the Wayback Machine
- LLM-optimized output formatting with text extraction
- Comprehensive error handling and logging

Main functions:
- mcp_list_archived_versions: List available snapshots for a URL
- mcp_get_archived_content: Fetch content from a specific archived version
- mcp_get_wayback_capabilities: Get service capabilities information
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from waybackpy import WaybackMachineCDXServerAPI

from mas_arena.mcp_collections.base import ActionArguments, ActionResponse, ActionCollection


class ArchivedVersion(BaseModel):
    """Individual archived version with structured data."""

    timestamp: str
    url: str
    status_code: str
    digest: str
    length: str
    mime_type: str


class WaybackMetadata(BaseModel):
    """Metadata for Wayback Machine operation results."""

    url: str
    operation: str  # 'list_versions', 'get_content', 'save_page'
    timestamp: str | None = None
    total_versions: int | None = None
    date_range: dict[str, str | None] | None = None
    content_length: int | None = None
    text_extracted: bool = False
    truncated: bool = False
    execution_time: float | None = None
    error_type: str | None = None
    user_agent: str = "MASARENA"


class WaybackActionCollection(ActionCollection):
    """MCP service for Wayback Machine operations.

    Provides comprehensive Wayback Machine functionality including:
    - Listing archived versions of URLs with flexible filtering
    - Fetching content from specific archived snapshots
    - LLM-friendly content formatting and text extraction
    - Robust error handling and detailed logging
    - Multiple output formats (markdown, JSON, text)
    """
    tool_name = "wayback"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Configuration
        self.user_agent = "MASARENA"
        self.default_timeout = 30
        self.max_content_length = 8192

    def _format_versions_for_llm(self, versions: list[ArchivedVersion], query_info: dict) -> str:
        """Format archived versions list for LLM consumption.

        Args:
            versions: List of archived versions
            query_info: Query information including URL and filters

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not versions:
            return f"No archived versions found for URL: {query_info.get('url', 'Unknown')}"

        output_parts = [
            f"# Wayback Machine Archives for {query_info.get('url', 'Unknown')}",
            f"\nFound **{len(versions)}** archived versions",
        ]

        if query_info.get("from_date") or query_info.get("to_date"):
            date_filter = []
            if query_info.get("from_date"):
                date_filter.append(f"From: {query_info['from_date']}")
            if query_info.get("to_date"):
                date_filter.append(f"To: {query_info['to_date']}")
            output_parts.append(f"\n**Date Filter:** {' | '.join(date_filter)}")

        output_parts.append("\n## Available Versions:")

        for i, version in enumerate(versions[:10], 1):  # Show first 10
            timestamp_formatted = self._format_timestamp(version.timestamp)
            output_parts.append(
                f"\n{i}. **{timestamp_formatted}**\n"
                f"   - Archive URL: {version.url}\n"
                f"   - Status: {version.status_code} | Size: {version.length} bytes\n"
                f"   - Type: {version.mime_type}"
            )

        if len(versions) > 10:
            output_parts.append(f"\n... and {len(versions) - 10} more versions")

        return "\n".join(output_parts)

    def _format_content_for_llm(self, content_data: dict, output_format: str = "markdown") -> str:
        """Format archived content for LLM consumption.

        Args:
            content_data: Content data dictionary
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if output_format == "json":
            return json.dumps(content_data, indent=2)

        elif output_format == "text":
            return content_data.get("content", "")

        else:  # markdown (default)
            output_parts = [
                f"# Archived Content from {content_data.get('url', 'Unknown')}",
                f"\n**Requested Timestamp:** {content_data.get('timestamp', 'Unknown')}",
                f"**Actual Timestamp:** {self._format_timestamp(content_data.get('fetched_timestamp', ''))}",
                f"**Content Length:** {content_data.get('original_content_length', 0):,} characters",
            ]

            if content_data.get("truncated"):
                output_parts.append(f"**Note:** Content truncated to {self.max_content_length:,} characters")

            if content_data.get("extract_text_only"):
                output_parts.append("**Note:** Text-only extraction applied")

            output_parts.extend(["\n## Content:", "\n---\n", content_data.get("content", ""), "\n---"])

            return "\n".join(output_parts)

    def _format_timestamp(self, timestamp: str) -> str:
        """Format Wayback Machine timestamp to human-readable format.

        Args:
            timestamp: Wayback timestamp (YYYYMMDDhhmmss)

        Returns:
            Human-readable timestamp
        """
        try:
            if len(timestamp) >= 14:
                dt = datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            return timestamp
        except (ValueError, TypeError):
            return timestamp or "Unknown"

    def _validate_wayback_parameters(self, url: str, timestamp: str = None) -> tuple[str, str | None]:
        """Validate and normalize Wayback Machine parameters.

        Args:
            url: URL to validate
            timestamp: Optional timestamp to validate

        Returns:
            Tuple of (validated_url, validated_timestamp)

        Raises:
            ValueError: If parameters are invalid
        """
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        validated_timestamp = None
        if timestamp:
            timestamp = timestamp.strip()
            if len(timestamp) < 8:
                raise ValueError("Timestamp must be at least 8 characters (YYYYMMDD)")
            validated_timestamp = timestamp

        return url, validated_timestamp

    async def mcp_list_archived_versions(
        self,
        url: str = Field(description="The URL of the website to check for archived versions"),
        limit: int = Field(default=10, description="Maximum number of versions to return (0 for all)"),
        from_date: str | None = Field(default=None, description="Start date filter (YYYYMMDDhhmmss)"),
        to_date: str | None = Field(default=None, description="End date filter (YYYYMMDDhhmmss)"),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """List available archived versions of a URL from the Wayback Machine."""
        start_time = time.time()

        # Handle FieldInfo objects
        if isinstance(url, FieldInfo):
            url = url.default
        if isinstance(limit, FieldInfo):
            limit = limit.default
        if isinstance(from_date, FieldInfo):
            from_date = from_date.default
        if isinstance(to_date, FieldInfo):
            to_date = to_date.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate parameters
            url, _ = self._validate_wayback_parameters(url)

            print(f"Listing archived versions for: {url}")

            # Query Wayback Machine CDX API
            cdx_api = WaybackMachineCDXServerAPI(url, user_agent=self.user_agent)
            all_snapshots = list(cdx_api.snapshots())

            # Apply date filtering
            if from_date or to_date:
                snapshots = [
                    s
                    for s in all_snapshots
                    if (not from_date or s.timestamp >= from_date) and (not to_date or s.timestamp <= to_date)
                ]
            else:
                snapshots = all_snapshots

            if not snapshots:
                return ActionResponse(
                    success=False,
                    message="No archived versions found for the specified URL and date range.",
                    metadata=WaybackMetadata(
                        url=url,
                        operation="list_versions",
                        total_versions=0,
                        date_range={"from_date": from_date, "to_date": to_date},
                        execution_time=time.time() - start_time,
                        error_type="no_results",
                    ).model_dump(),
                )

            # Convert to structured format
            versions = [
                ArchivedVersion(
                    timestamp=snapshot.timestamp,
                    url=snapshot.archive_url,
                    status_code=snapshot.statuscode,
                    digest=snapshot.digest,
                    length=snapshot.length,
                    mime_type=snapshot.mimetype,
                )
                for snapshot in snapshots
            ]

            # Apply limit
            if limit > 0 and len(versions) > limit:
                versions = versions[:limit]

            # Format output
            query_info = {"url": url, "from_date": from_date, "to_date": to_date, "total_found": len(snapshots)}

            if output_format == "json":
                message = [version.model_dump() for version in versions]
            else:
                message = self._format_versions_for_llm(versions, query_info)

            execution_time = time.time() - start_time
            print(f"Found {len(versions)} archived versions in {execution_time:.2f}s")

            return ActionResponse(
                success=True,
                message=message,
                metadata=WaybackMetadata(
                    url=url,
                    operation="list_versions",
                    total_versions=len(snapshots),
                    date_range={"from_date": from_date, "to_date": to_date},
                    execution_time=execution_time,
                ).model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to list archived versions: {str(e)}"
            print(f"❌ Error in mcp_list_archived_versions: {str(e)}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WaybackMetadata(
                    url=url or "unknown",
                    operation="list_versions",
                    execution_time=time.time() - start_time,
                    error_type=type(e).__name__,
                ).model_dump(),
            )

    async def mcp_get_archived_content(
        self,
        url: str = Field(description="The URL of the website to fetch archived content from"),
        timestamp: str = Field(description="The timestamp of the desired version (YYYYMMDDhhmmss)"),
        extract_text_only: bool = Field(default=True, description="Extract only text content, removing HTML tags"),
        truncate_content: bool = Field(default=False, description="Truncate content to manageable length for LLMs"),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Fetch content from a specific archived page version."""
        start_time = time.time()

        # Handle FieldInfo objects
        if isinstance(url, FieldInfo):
            url = url.default
        if isinstance(timestamp, FieldInfo):
            timestamp = timestamp.default
        if isinstance(extract_text_only, FieldInfo):
            extract_text_only = extract_text_only.default
        if isinstance(truncate_content, FieldInfo):
            truncate_content = truncate_content.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Validate parameters
            url, timestamp = self._validate_wayback_parameters(url, timestamp)

            print(f"Fetching archived content: {url} at {timestamp}")

            # Query Wayback Machine for closest snapshot
            cdx_api = WaybackMachineCDXServerAPI(url, user_agent=self.user_agent)
            snapshot = cdx_api.near(wayback_machine_timestamp=timestamp)

            if not snapshot or not snapshot.archive_url:
                return ActionResponse(
                    success=False,
                    message=f"No archived version found for {url} at timestamp {timestamp}",
                    metadata=WaybackMetadata(
                        url=url,
                        operation="get_content",
                        timestamp=timestamp,
                        execution_time=time.time() - start_time,
                        error_type="no_snapshot",
                    ).model_dump(),
                )

            # Fetch content
            response = requests.get(snapshot.archive_url, timeout=self.default_timeout)
            response.raise_for_status()
            content = response.text
            original_length = len(content)

            # Extract text if requested
            if extract_text_only:
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text(separator=" ", strip=True)

            # Truncate if requested
            truncated = False
            if truncate_content and len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "..."
                truncated = True

            # Prepare content data
            content_data = {
                "url": url,
                "timestamp": timestamp,
                "fetched_timestamp": snapshot.timestamp,
                "content": content,
                "original_content_length": original_length,
                "truncated": truncated,
                "extract_text_only": extract_text_only,
            }

            # Format output
            if output_format == "json":
                message = content_data
            elif output_format == "text":
                message = content
            else:  # markdown
                message = self._format_content_for_llm(content_data, output_format)

            execution_time = time.time() - start_time
            print(f"✅ Retrieved {len(content):,} characters in {execution_time:.2f}s")

            return ActionResponse(
                success=True,
                message=message,
                metadata=WaybackMetadata(
                    url=url,
                    operation="get_content",
                    timestamp=snapshot.timestamp,
                    content_length=len(content),
                    text_extracted=extract_text_only,
                    truncated=truncated,
                    execution_time=execution_time,
                ).model_dump(),
            )

        except Exception as e:
            error_msg = f"Failed to fetch archived content: {str(e)}"
            print(f"❌ Error in mcp_get_archived_content: {str(e)}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=WaybackMetadata(
                    url=url or "unknown",
                    operation="get_content",
                    timestamp=timestamp or "unknown",
                    execution_time=time.time() - start_time,
                    error_type=type(e).__name__,
                ).model_dump(),
            )

    def mcp_get_wayback_capabilities(self) -> ActionResponse:
        """Get Wayback Machine service capabilities and configuration."""
        capabilities = {
            "service": "Wayback Machine MCP Server",
            "version": "1.0.0",
            "description": "Interact with the Internet Archive's Wayback Machine",
            "operations": {
                "list_versions": "List archived versions of URLs with date filtering",
                "get_content": "Fetch content from specific archived snapshots",
            },
            "features": {
                "date_filtering": True,
                "text_extraction": True,
                "content_truncation": True,
                "multiple_formats": ["markdown", "json", "text"],
                "error_handling": True,
                "logging": True,
            },
            "configuration": {
                "user_agent": self.user_agent,
                "default_timeout": self.default_timeout,
                "max_content_length": self.max_content_length,
            },
            "limits": {"max_content_length": self.max_content_length, "request_timeout": self.default_timeout},
        }

        message = f"""# Wayback Machine Service Capabilities

        ## Service Information
        - **Service:** {capabilities["service"]}
        - **Version:** {capabilities["version"]}
        - **Description:** {capabilities["description"]}

        ## Available Operations
        - **List Versions:** {capabilities["operations"]["list_versions"]}
        - **Get Content:** {capabilities["operations"]["get_content"]}


        ## Features
        - **Date Filtering:** {capabilities["features"]["date_filtering"]}
        - **Text Extraction:** {capabilities["features"]["text_extraction"]}
        - **Content Truncation:** {capabilities["features"]["content_truncation"]}
        - **Output Formats:** {", ".join(capabilities["features"]["multiple_formats"])}
        - **Error Handling:** {capabilities["features"]["error_handling"]}
        - **Logging:** {capabilities["features"]["logging"]}

        ## Configuration
        - **User Agent:** {capabilities["configuration"]["user_agent"]}
        - **Default Timeout:** {capabilities["configuration"]["default_timeout"]} seconds
        - **Max Content Length:** {capabilities["configuration"]["max_content_length"]:,} characters

        ## Limits
        - **Max Content Length:** {capabilities["limits"]["max_content_length"]:,} characters
        - **Request Timeout:** {capabilities["limits"]["request_timeout"]} seconds
        """

        return ActionResponse(success=True, message=message, metadata=capabilities)


# Default arguments for testing
if __name__ == "__main__":
    import sys
    import json
    
    # Determine if running in MCP tool mode (called without arguments)
    is_mcp_mode = len(sys.argv) == 1
    if is_mcp_mode:
        original_print = print
        print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
    
    load_dotenv()
    
    arguments = ActionArguments(
        name="wayback-machine-server",
        transport="stdio",
        workspace=os.getenv("MASARENA_WORKSPACE", "~"),
    )
    
    try:
        service = WaybackActionCollection(arguments)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                function_name = input_data.get("function_name", input_data.get("name", ""))
                arguments = input_data.get("arguments", {})
                
                if function_name == "mcp_list_archived_versions":
                    result = asyncio.run(service.mcp_list_archived_versions(
                        url=arguments.get("url", ""),
                        limit=arguments.get("limit", 10),
                        from_date=arguments.get("from_date", None),
                        to_date=arguments.get("to_date", None),
                        output_format=arguments.get("output_format", "markdown")
                    ))
                elif function_name == "mcp_get_archived_content":
                    result = asyncio.run(service.mcp_get_archived_content(
                        url=arguments.get("url", ""),
                        timestamp=arguments.get("timestamp", ""),
                        extract_text_only=arguments.get("extract_text_only", True),
                        truncate_content=arguments.get("truncate_content", False),
                        output_format=arguments.get("output_format", "markdown")
                    ))
                elif function_name == "mcp_get_wayback_capabilities":
                    result = service.mcp_get_wayback_capabilities()
                else:
                    result = ActionResponse(
                        success=False,
                        message=f"Unknown function: {function_name}",
                        metadata={"error_type": "unknown_function"}
                    )
                
                # Write result to stdout as JSON
                sys.stdout.write(json.dumps(result.model_dump()) + "\n")
                sys.stdout.flush()
                sys.exit(0)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error parsing input JSON: {str(e)}\n")
                sys.exit(1)
            except Exception as e:
                sys.stderr.write(f"Error processing request: {str(e)}\n{traceback.format_exc()}\n")
                sys.exit(1)
        else:
            service.run()
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
        sys.exit(1)
