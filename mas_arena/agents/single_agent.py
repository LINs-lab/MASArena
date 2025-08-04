"""
Single Agent System

This module implements a simple single-agent system that uses a single LLM
to solve problems directly.
"""

import os
from typing import Dict, Any, List
import contextlib
import json

from openai import AsyncOpenAI
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SingleAgent(AgentSystem):
    """
    Single Agent System

    This agent system uses a single LLM to solve problems directly.
    """

    def __init__(self, name: str = "single_agent", config: Dict[str, Any] = None, mcp_config: Dict[str, Any] = None):
        """Initialize the Single Agent System"""
        super().__init__(name, config, mcp_config)
        self.config = config or {}

        # Default model name can be overridden by config
        self.model_name = self.config.get("llm_config", {}).get("model") or os.getenv("MODEL_NAME", "gpt-4-1106-preview")

        # System prompt can be customized
        self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt

        # Agent's state
        self.max_steps = self.config.get("max_steps", 10)
        self.current_step = 0
        self.message_history = []
        self.tools = [] # This will be populated by the ToolIntegrationWrapper

        # Set up the OpenAI client
        llm_config = self.config.get("llm_config", {})
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = llm_config.get("api_base") or os.getenv("OPENAI_API_BASE")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def prepare_llm_input(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepares the input for the LLM, including transforming tools to the expected format."""
        llm_input = {"messages": messages}
        if tools:
            llm_tools = []
            for tool in tools:
                # Reconstruct the tool format expected by OpenAI.
                # Our internal tool representation has function_name, description, etc. at the top level.
                function_spec = {
                    "name": tool.get("function_name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters")
                }
                
                # Check if we have the essential parts for a valid tool definition.
                if function_spec["name"] and function_spec["description"] and function_spec["parameters"]:
                    llm_tools.append({
                        "type": "function",
                        "function": function_spec
                    })

            if llm_tools:
                llm_input["tools"] = llm_tools
                llm_input["tool_choice"] = "auto"
        return llm_input

    def extract_tool_calls(self, response: Any) -> (bool, List[Dict[str, Any]]):
        """Extracts tool calls from the LLM response."""
        if not response.choices or not response.choices[0].message.tool_calls:
            return False, []

        tool_calls = []
        for tc in response.choices[0].message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        return True, tool_calls

    async def execute_tool(self, tool_name: str, tool_args: str) -> Dict[str, Any]:
        """Executes a tool call."""
        # Find the tool details from the original tool list
        tool_details = next((t for t in self.tools if t.get("function_name") == tool_name), None)
        if not tool_details:
            return {"error": f"Tool '{tool_name}' not found."}

        server_name = tool_details.get("server_name")
        function_name = tool_details.get("function_name")

        if not server_name or not function_name:
            return {"error": f"Invalid tool configuration for '{tool_name}'."}

        try:
            arguments = json.loads(tool_args)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON arguments for tool '{tool_name}': {tool_args}"}

        # Use the ToolManager to call the tool
        return await self.tool_manager.call_tool(server_name, function_name, arguments)

    def parse_generated_text(self, generated_text: str, parser: Any, parse_mode: str) -> Any:
        """
        Parse the generated text using the provided parser.
        """
        with contextlib.suppress(Exception):
            if parse_mode == "str":
                return parser.parse(text=generated_text)
            elif parse_mode == "json":
                return parser.parse(json_str=generated_text)
        return generated_text


    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.

        This method implements the actual agent logic without handling evaluation or metrics.

        Args:
            problem: Dictionary containing the problem data

        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]

        self.current_step = 0
        self.message_history = []

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem_text}"},
        ]
        self.message_history = messages.copy()

        all_messages = []
        final_answer = None

        while self.current_step < self.max_steps:
            self.current_step += 1

            llm_input = self.prepare_llm_input(self.message_history, self.tools)
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=llm_input["messages"],
                    tools=llm_input.get("tools", None),
                    tool_choice=llm_input.get("tool_choice", None)
                )
            except Exception as e:
                print(f"call llm_input: {llm_input}")
                print(e)

            response_content = response.choices[0].message.content
            response_content = response_content.replace('\r\n', '\n').replace('\r',
                                                                              '\n').strip() if response_content else ""

            ai_message = {
                'content': response_content,
                'name': 'single_agent',
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': response.usage
            }

            has_tool_call, tool_calls = self.extract_tool_calls(response)

            if has_tool_call and tool_calls:
                ai_message['tool_calls'] = tool_calls

                self.message_history.append({
                    "role": "assistant",
                    "content": response_content if response_content else None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    } for tc in tool_calls]
                })

                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    tool_id = tool_call["id"]

                    tool_result = await self.execute_tool(tool_name, tool_args)

                    self.message_history.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": json.dumps(tool_result)
                    })

                    tool_message = {
                        'name': f'tool_{tool_name}',
                        'role': 'tool',
                        'content': json.dumps(tool_result),
                        'tool_call_id': tool_id,
                        'message_type': 'tool_response'
                    }
                    all_messages.append(tool_message)
            else:
                final_answer = response_content

                self.message_history.append({
                    "role": "assistant",
                    "content": response_content
                })

                break

            all_messages.append(ai_message)

        if final_answer is None and all_messages:
            last_message = all_messages[-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                final_answer = last_message['content']
            else:
                final_answer = "No final answer provided after reaching maximum steps."

        if self.message_history and self.message_history[-1]["role"] == "assistant":
            last_ai_message = {
                'content': self.message_history[-1]["content"],
                'name': 'single_agent',
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': response.usage
            }
            all_messages.append(last_ai_message)

        if "parser" in kwargs or "parse_mode" in kwargs:
            parser = kwargs.get("parser", None)
            parse_mode = kwargs.get("parse_mode", "str")
            response_format = self.parse_generated_text(final_answer, parser=parser, parse_mode=parse_mode)
            return {
                "messages": all_messages,
                "final_answer": response_format
            }

        # Return the response and message with usage metadata for the evaluate method
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }


# Register the agent system
AgentSystemRegistry.register("single_agent", SingleAgent)
