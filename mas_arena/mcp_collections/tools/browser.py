"""
Browser MCP Server

This module provides MCP server functionality for browser automation and interaction.
It handles tasks such as web scraping, form submission, and automated browsing using browser-use package.

Main functions:
- mcp_browser_use: Performs browser automation tasks with LLM-friendly output
"""

import json
import os
import re
import sys
import time
import traceback
import logging
from pathlib import Path

try:
    from browser_use import Agent, AgentHistoryList, BrowserProfile
    from browser_use.llm import ChatOpenAI
    from dotenv import load_dotenv
    from pydantic import BaseModel, Field


    from ..base import ActionArguments, ActionCollection, ActionResponse
except Exception as e:
    sys.stderr.write(f"Failed to import browser tool: {traceback.format_exc()}\n")
    raise e

# Suppress browser_use library logging
logging.getLogger("browser_use").setLevel(logging.CRITICAL)





class BrowserMetadata(BaseModel):
    """Metadata for browser automation results."""

    task: str
    execution_successful: bool
    steps_taken: int | None = None
    downloaded_files: list[str] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    execution_time: float | None = None
    error_type: str | None = None
    trace_log_path: str | None = None


class BrowserActionCollection(ActionCollection):
    """MCP service for browser automation using browser-use package.

    Provides comprehensive web automation capabilities including:
    - Web scraping and content extraction
    - Form submission and interaction
    - File downloads and media handling
    - LLM-enhanced browsing with memory
    - Robot detection and paywall handling
    """
    tool_name = "browser"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()

        # Extended system prompt for browser automation
        self.extended_browser_system_prompt = """
10. URL ends with .pdf
- If the go_to_url function with `https://any_url/any_file_name.pdf` as the parameter, just report the url link and hint the user to download using `download` mcp tool or `curl`, then execute `done` action.

11. Robot Detection:
- If the page is a robot detection page, abort immediately. Then navigate to the most authoritative source for similar information instead

# Efficiency Guidelines
0. if download option is available, always **DOWNLOAD** as possible! Also, report the download url link in your result.
1. Use specific search queries with key terms from the task
2. Avoid getting distracted by tangential information
3. If blocked by paywalls, try archive.org or similar alternatives
4. Document each significant finding clearly and concisely
5. Precisely extract the necessary information with minimal browsing steps.
"""

        # Initialize LLM configuration
        self.llm_config = ChatOpenAI(
            model=os.getenv("LLM_MODEL_NAME"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=1.0,
        )
        
        # Log LLM config
        print(f"Browser llm_config: {self.llm_config}")

        # Initialize browser profile
        self.workspace = Path(os.path.expanduser(arguments.workspace))
        downloads_dir = os.getenv("BROWSER_DOWNLOADS_DIR", str(self.workspace / "downloads"))
        self.browser_profile = BrowserProfile(
            downloads_path=downloads_dir,
            cookies_enabled=True,
            cache_path=str(self.workspace / "browser_cache"),
        )

        # Set up trace logging
        self.trace_log_dir = os.getenv("BROWSER_TRACE_LOG_DIR", str(self.workspace / "browser_logs"))
        os.makedirs(self.trace_log_dir, exist_ok=True)
        
        # Log initialization details
        print("Browser automation service initialized")
        print(f"Downloads directory: {self.browser_profile.downloads_path}")
        print(f"Trace logs directory: {self.trace_log_dir}")

    def _create_browser_agent(self, task: str) -> Agent:
        """Create a browser agent instance with configured settings.

        Args:
            task: The task description for the browser agent

        Returns:
            Configured Agent instance
        """
        os.environ["TELEMETRY_ENABLED"] = "false"
        return Agent(
            task=task,
            llm=self.llm_config,
            extend_system_message=self.extended_browser_system_prompt,
            use_vision=True,
            enable_memory=False,
            browser_profile=self.browser_profile,
            save_conversation_path=f"{self.trace_log_dir}/trace.log",
            telemetry_enabled=False,
        )

    def _extract_visited_urls(self, extracted_content: list[str]) -> list[str]:
        """Inner method to extract URLs from content using regex.

        Args:
            content_list: List of content strings to search for URLs

        Returns:
            List of unique URLs found in the content
        """
        url_pattern = r'https?://[^\s<>"\[\]{}|\\^`]+'
        visited_urls = set()

        for content in extracted_content:
            if content and isinstance(content, str):
                urls = re.findall(url_pattern, content)
                visited_urls.update(urls)

        return list(visited_urls)

    def _format_extracted_content(self, extracted_content: list[str]) -> str:
        """Format extracted content to be LLM-friendly.

        Args:
            extracted_content: List of extracted content strings from browser execution

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not extracted_content:
            return "No content extracted from browser execution."

        # Handle list of strings
        if len(extracted_content) == 1:
            # Single item - return it directly with formatting
            return f"**Extracted Content:**\n{extracted_content[0]}"
        else:
            # Multiple items - format as numbered list
            formatted_parts = ["**Extracted Content:**"]
            for i, content in enumerate(extracted_content, 1):
                if content.strip():  # Only include non-empty content
                    formatted_parts.append(f"{i}. {content}")

            return (
                "\n".join(formatted_parts)
                if len(formatted_parts) > 1
                else "No meaningful content extracted from browser execution."
            )

    async def mcp_browser_use(
        self,
        task: str = Field(
            description="The task to perform using the browser automation agent"
        ),
        max_steps: int = Field(
            default=50, description="Maximum number of steps for browser execution"
        ),
        extract_format: str = Field(
            default="markdown",
            description="Format for extracted content: 'markdown', 'json', or 'text'",
        ),
    ) -> ActionResponse:
        """Perform browser automation tasks using the browser-use package.

        This tool provides comprehensive browser automation capabilities including:
        - Web scraping and content extraction
        - Form submission and automated interactions
        - File downloads and media handling
        - LLM-enhanced browsing with memory and vision
        - Automatic handling of robot detection and paywalls
        """
        try:
            print(f"🎯 Starting browser task: {task}")

            # Create browser agent
            agent = self._create_browser_agent(task)

            start_time = time.time()

            browser_execution: AgentHistoryList = await agent.run(max_steps=max_steps)

            # Poll for completion
            while not browser_execution.is_done():
                await asyncio.sleep(1)


            execution_time = time.time() - start_time

            if (
                browser_execution.is_done()
                and browser_execution.is_successful()
            ):
                # Extract and format content
                extracted_content = browser_execution.extracted_content()
                final_result = browser_execution.final_result()

                # Format content based on requested format
                if extract_format.lower() == "json":
                    formatted_content = json.dumps(
                        {"summary": final_result, "extracted_data": extracted_content},
                        indent=2,
                    )
                elif extract_format.lower() == "text":
                    formatted_content = f"{final_result}\n\n{self._format_extracted_content(extracted_content)}"
                else:  # markdown (default)
                    formatted_content = (
                        f"## Browser Automation Result\n\n**Summary:** {final_result}\n\n"
                        f"{self._format_extracted_content(extracted_content)}"
                    )

                # Prepare metadata
                metadata = BrowserMetadata(
                    task=task,
                    execution_successful=True,
                    steps_taken=(
                        len(browser_execution.history)
                        if hasattr(browser_execution, "history")
                        else None
                    ),
                    downloaded_files=[],
                    visited_urls=self._extract_visited_urls(extracted_content),
                    execution_time=execution_time,
                    trace_log_path=f"{self.trace_log_dir}/trace.log",
                )

                print(f"🗒️ Detail: {extracted_content}")
                print(f"🌏 Result: {final_result}")

                return ActionResponse(
                    success=True,
                    message=formatted_content,
                    metadata=metadata.model_dump(),
                )

            else:
                # Handle execution failure
                error_msg = "Browser execution failed or was not completed successfully"

                metadata = BrowserMetadata(
                    task=task,
                    execution_successful=False,
                    execution_time=execution_time,
                    error_type="execution_failure",
                    trace_log_path=f"{self.trace_log_dir}/trace.log",
                )

                print(f"❌ {error_msg}")

                return ActionResponse(
                    success=False, message=error_msg, metadata=metadata.model_dump()
                )

        except Exception as e:
            error_msg = f"Error during browser automation: {str(e)}"
            self.logger.error(f"{error_msg}")
            self.logger.error(traceback.format_exc())

            metadata = BrowserMetadata(
                task=task,
                execution_successful=False,
                execution_time=None,
                error_type="exception",
            )

            print(f"❌ {error_msg}")

            return ActionResponse(
                success=False,
                message=f"Browser automation error: {str(e)}",
                metadata=metadata.model_dump(),
            )

    def mcp_get_browser_capabilities(self) -> ActionResponse:
        """Get information about browser automation capabilities and configuration.

        Returns:
            ActionResponse with browser service capabilities and current configuration
        """
        capabilities = {
            "automation_features": [
                "Web scraping and content extraction",
                "Form submission and interaction",
                "File downloads and media handling",
                "LLM-enhanced browsing with vision",
                "Memory-enabled browsing sessions",
                "Robot detection and paywall handling",
            ],
            "supported_formats": ["markdown", "json", "text"],
            "configuration": {
                "llm_model": os.getenv("LLM_MODEL_NAME", "Not configured"),
                "downloads_directory": self.browser_profile.downloads_path,
                "cookies_enabled": bool(os.getenv("COOKIES_FILE_PATH")),
                "trace_logging": True,
                "vision_enabled": True,
                "headless": True,
            },
        }

        formatted_info = f"""# Browser Automation Service Capabilities

        ## Features
        {chr(10).join(f"- {feature}" for feature in capabilities["automation_features"])}

        ## Supported Output Formats
        {chr(10).join(f"- {fmt}" for fmt in capabilities["supported_formats"])}

        ## Current Configuration
        - **LLM Model:** {capabilities["configuration"]["llm_model"]}
        - **Downloads Directory:** {capabilities["configuration"]["downloads_directory"]}
        - **Cookies Enabled:** {capabilities["configuration"]["cookies_enabled"]}
        - **Vision Enabled:** {capabilities["configuration"]["vision_enabled"]}
        - **Memory Enabled:** {capabilities["configuration"]["memory_enabled"]}
        - **Trace Logging:** {capabilities["configuration"]["trace_logging"]}
        """

        return ActionResponse(
            success=True, message=formatted_info, metadata=capabilities
        )


# Example usage and entry point
if __name__ == "__main__":
    import sys
    import json
    import asyncio
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        load_dotenv()
        
        test_task = "Visit https://example.com and extract the main content"
        test_max_steps = 10
        test_format = "markdown"
        
        # Create test instance
        args = ActionArguments(
            name="browser_automation_service",
            transport="stdio"
        )
        
        # Run test
        service = BrowserActionCollection(args)
        result = asyncio.run(service.mcp_browser_use(test_task, test_max_steps, test_format))
        print(json.dumps(result.model_dump()))
        sys.exit(0)
    
    # Standard MCP server mode
    is_mcp_mode = len(sys.argv) == 1
    if is_mcp_mode:
        # Shut down any existing logging handlers to ensure clean stdout
        logging.shutdown()
        
        original_print = print
        # Redirect print to stderr to avoid interfering with stdout
        print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
    
    # Disable telemetry through environment variable
    os.environ["TELEMETRY_ENABLED"] = "false"
    
    load_dotenv()
    
    args = ActionArguments(
        name="browser_automation_service",
        transport="stdio",
        workspace=os.getenv("MASARENA_WORKSPACE", "~")
    )
    
    try:
        service = BrowserActionCollection(args)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                function_name = input_data.get("function_name", input_data.get("name", "browser_use"))
                arguments = input_data.get("arguments", {})
                
                if function_name ==  "mcp_browser_use":
                    result = asyncio.run(service.mcp_browser_use(
                        task=arguments.get("task", ""),
                        max_steps=arguments.get("max_steps", 10),
                        extract_format=arguments.get("extract_format", "markdown")
                    ))
                elif function_name == "get_browser_capabilities":
                    result = service.mcp_get_browser_capabilities()
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
