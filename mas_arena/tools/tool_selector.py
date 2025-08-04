from typing import Any, List, Dict, Optional
import json
import logging
from openai import AsyncOpenAI

# Set up a logger for the tool selector
logger = logging.getLogger(__name__)

LLM_TOOL_SELECTOR_PROMPT = """
You are an expert at selecting the most relevant tools for a given task.
The user will provide a task description and a list of available tools. Your job is to identify the tools that are most likely to be necessary to complete the task.

The user's task is:
---
{task_description}
---

Here is a list of available tools with their descriptions:
---
{tool_summaries}
---

Please analyze the task and the tools. Respond with a JSON-formatted list containing the exact `function_name` of the tools you believe are necessary.
For example:
["mcp_read_file", "mcp_web_search", "mcp_run_python_code"]

If you think no tools are needed, return an empty list [].
Only return the JSON list, with no other text before or after it.
"""

class ToolSelector:
    """
    Selects tools for a given task using an LLM.
    """

    def __init__(self, tools: List[Dict[str, Any]], llm_client: AsyncOpenAI):
        """
        Initialize with available tools and an LLM client.
        
        Args:
            tools: List of full tool definitions from ToolManager.
            llm_client: An asynchronous OpenAI client instance.
        """
        self.tools = tools
        self.tool_map = {
            tool.get("function_name", tool.get("name")): tool for tool in tools
        }
        self.llm_client = llm_client
        logger.info(f"ToolSelector initialized with {len(self.tools)} tools.")

    def _create_tool_summaries(self) -> str:
        """Creates a simplified string representation of tools for the LLM prompt."""
        summaries = []
        for tool in self.tools:
            # Prefer function_name as the primary identifier
            name = tool.get("function_name", tool.get("name"))
            description = tool.get("description", "No description available.")
            summaries.append(f"- function_name: {name}\n  description: {description}")
        return "\n".join(summaries)

    async def _select_with_llm(self, task_description: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Uses an LLM to select the most relevant tools for a task.
        
        Args:
            task_description: The description of the task to be performed.
            limit: A hard limit on the number of tools to return, to prevent misuse.

        Returns:
            A list of full tool definitions for the selected tools.
        """
        if not self.llm_client:
            logger.error("LLM client is not available. Cannot perform tool selection.")
            return []

        tool_summaries = self._create_tool_summaries()
        prompt = LLM_TOOL_SELECTOR_PROMPT.format(
            task_description=task_description,
            tool_summaries=tool_summaries
        )

        try:
            logger.info("Asking LLM to select tools for the task...")
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned an empty response for tool selection.")
                return []
                
            # Clean up the response to get only the JSON part
            json_str = content.strip().replace("```json", "").replace("```", "").strip()
            
            selected_tool_names = json.loads(json_str)
            logger.info(f"LLM selected the following tools: {selected_tool_names}")

            if not isinstance(selected_tool_names, list):
                logger.warning(f"LLM returned a non-list for tool selection: {selected_tool_names}")
                return []

            # Retrieve full tool definitions based on the selected names
            selected_tools = [
                self.tool_map[name]
                for name in selected_tool_names
                if name in self.tool_map
            ]
            
            # Apply the hard limit
            return selected_tools[:limit]

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {content}", exc_info=True)
            return []
        except Exception:
            logger.error("An unexpected error occurred during LLM-based tool selection.", exc_info=True)
            return []

    async def select_tools(
        self,
        task_description: str,
        num_agents: Optional[int] = None,
        limit: int = 10
    ) -> Any:
        """
        Unified public interface for tool selection.
        This implementation uses an LLM for selection.
        """
        # For now, multi-agent partitioning will also use the same flat list of selected tools.
        # A more advanced implementation could partition them.
        selected_tools = await self._select_with_llm(task_description, limit)
        
        if num_agents and num_agents > 1:
            # Basic round-robin partitioning for multi-agent scenarios
            partitions: List[List[Dict[str, Any]]] = [[] for _ in range(num_agents)]
            if not selected_tools:
                return partitions
            for i, tool in enumerate(selected_tools):
                partitions[i % num_agents].append(tool)
            return partitions
            
        return selected_tools
