from typing import Optional
import os
import asyncio
import requests
from datetime import datetime

from smolagents import Tool
from smolagents.models import Model

from browser_use import Agent, Browser, ChatOpenAI, Controller, Tools

from mas_arena.utils.anchor_keys import get_anchor_api_keys, is_anchor_quota_error
from mas_arena.utils.env import get_openai_api_base, get_openai_api_key


class BrowserTool(Tool):
    name = "browse_web"
    description = """
Use this tool to browse the web and gather information from websites. 
This tool can navigate websites, search for information, fill forms, click buttons, and extract data.
It uses an AI agent to control a remote browser and perform complex web interactions based on your description."""

    inputs = {
        "task_description": {
            "description": "A clear description of what you want to accomplish on the web. Be specific about what information you need or what actions to perform. Examples: 'Search for latest news about AI on Google', 'Find the price of iPhone 15 on Amazon', 'Compare features of different laptops on tech review sites'.",
            "type": "string",
        },
        "starting_url": {
            "description": "[Optional]: The URL to start browsing from. If not provided, the agent will determine the best starting point based on the task.",
            "type": "string",
            "nullable": True,
        },
        "max_steps": {
            "description": "[Optional]: Maximum number of steps the agent can take. Default is 20 to prevent long execution times.",
            "type": "integer",
            "nullable": True,
        },
        "use_vision": {
            "description": "[Optional]: Whether to use vision capabilities for better web interaction. Default is True. Set to False to reduce costs.",
            "type": "boolean",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int = 8000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.browser_llm_model = os.getenv("CHATGPT_MODEL")

    def _get_browser_llm(self):
        return ChatOpenAI(
            model=self.browser_llm_model,
            base_url=get_openai_api_base(),
            api_key=get_openai_api_key(),
        )

    def _create_remote_browser_session(self) -> str:
        anchor_api_keys = get_anchor_api_keys()
        if not anchor_api_keys:
            raise ValueError(
                "ANCHOR_API_KEY environment variable is required for remote browser access"
            )

        url = "https://api.anchorbrowser.io/v1/sessions"
        last_error: Exception | None = None

        for anchor_api_key in anchor_api_keys:
            headers = {
                "anchor-api-key": anchor_api_key,
                "Content-Type": "application/json",
            }
            try:
                response = requests.post(url, headers=headers)
                response.raise_for_status()
                cdp_url = response.json()["data"]["cdp_url"]
                return cdp_url
            except requests.RequestException as e:
                last_error = e
                if not is_anchor_quota_error(e):
                    raise Exception(f"Failed to create remote browser session: {str(e)}")

        if last_error:
            raise Exception(f"Failed to create remote browser session: {str(last_error)}")
        raise ValueError("Missing Anchor API key (ANCHOR_API_KEY).")

    def _format_history_summary(self, history) -> str:
        summary = []

        # Use the documented API methods for AgentHistoryList
        try:
            # Basic completion status
            summary.append(
                f"Browser automation completed successfully: {history.is_done()}"
            )
            summary.append(f"Total steps taken: {history.number_of_steps()}")

            # Errors encountered
            errors = history.errors()
            if errors:
                # Filter out None values
                actual_errors = [e for e in errors if e is not None]
                if actual_errors:
                    summary.append(f"\nErrors encountered: {len(actual_errors)}")
                    for error in actual_errors[:3]:
                        summary.append(f"  - {str(error)[:100]}...")

            # Final result
            final_result = history.final_result()
            if final_result:
                summary.append(f"\nExtracted information:")
                result_str = str(final_result)
                if len(result_str) > self.text_limit:
                    summary.append(f"{result_str[:self.text_limit]}...")
                else:
                    summary.append(result_str)

            # All extracted content (if different from final result)
            extracted_content = history.extracted_content()
            if extracted_content and len(extracted_content) > 1:
                summary.append(f"\nTotal content extractions: {len(extracted_content)}")

            # Success status
            success_status = history.is_successful()
            if success_status is not None:
                summary.append(f"\nTask completed successfully: {success_status}")

        except AttributeError as e:
            # Fallback for older versions or different object types
            summary.append(f"Could not parse history object: {str(e)}")
            summary.append(
                f"Available methods: {[method for method in dir(history) if not method.startswith('_')]}"
            )

            # Try to extract basic information
            if hasattr(history, "__str__"):
                summary.append(f"\nRaw result: {str(history)[:self.text_limit]}...")

        return "\n".join(summary)

    async def _run_browser_agent(
        self,
        task: str,
        starting_url: Optional[str] = None,
        max_steps: int = 50,
        use_vision: bool = True,
    ) -> str:
        try:
            cdp_url = self._create_remote_browser_session()

            llm = self._get_browser_llm()
            tools = Tools()

            browser = Browser(cdp_url=cdp_url, headless=False, keep_alive=False)

            if starting_url:
                full_task = f"Starting from {starting_url}, {task}"
            else:
                full_task = task

            controller = Controller()
            agent = Agent(
                task=full_task,
                llm=llm,
                tools=tools,
                browser=browser,
                controller=controller,
                enable_memory=False,
                use_vision=False,
            )

            result = await agent.run(max_steps=max_steps)

            return self._format_history_summary(result)

        except Exception as e:
            return f"Browser automation failed: {str(e)}\n\nPlease check your ANCHOR_API_KEY/ANCHOR_API_KEY_2 and OPENAI_API_KEY environment variables."

    def forward(
        self,
        task_description: str,
        starting_url: Optional[str] = None,
        max_steps: Optional[int] = None,
        use_vision: Optional[bool] = None,
    ) -> str:
        if max_steps is None:
            max_steps = 10
        if use_vision is None:
            use_vision = True

        if not task_description.strip():
            return "Error: Task description cannot be empty. Please provide a clear description of what you want to accomplish."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # The asyncio.run() function can cause issues when used with nest_asyncio,
            # as it tries to create a new event loop when one is already running.
            # This can lead to the process hanging indefinitely.
            # To fix this, we get the existing event loop (patched by nest_asyncio)
            # and run the async task within that loop. This ensures that the timeout
            # is respected and the program doesn't hang.
            loop = asyncio.get_event_loop()
            task = self._run_browser_agent(
                task_description, starting_url, max_steps, use_vision
            )
            future = asyncio.wait_for(task, timeout=300.0)
            result = loop.run_until_complete(future)

            output = [
                f"Browser Automation Report - {timestamp}",
                f"Task: {task_description}",
                f"Starting URL: {starting_url or 'Auto-determined'}",
                f"Max Steps: {max_steps}",
                f"Vision Enabled: {use_vision}",
                "=" * 50,
                result,
            ]

            return "\n".join(output)

        except asyncio.TimeoutError:
            return f"Browser automation timed out after 300 seconds."
        except Exception as e:
            error_msg = [
                f"Browser Automation Error - {timestamp}",
                f"Task: {task_description}",
                f"Error: {str(e)}",
                "",
                "Troubleshooting tips:",
                "1. Ensure you have set ANCHOR_API_KEY for remote browser access",
                "2. Ensure you have set OPENAI_API_KEY for LLM access",
                "3. Check your internet connection",
                "4. Try with a simpler task description",
                "5. Set use_vision=False to reduce resource usage",
            ]
            return "\n".join(error_msg)
