import json
import os
import sys
import time
import traceback
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import Field
from pydantic.fields import FieldInfo

from ..base import ActionArguments, ActionCollection, ActionResponse


class ThinkCollection(ActionCollection):
    """MCP service for complex problem reasoning using powerful reasoning models.

    Supports advanced reasoning for:
    - Mathematical problems and proofs
    - Code contest and programming challenges
    - Logic puzzles and riddles
    - Competition-level STEM problems
    - Multi-step analytical reasoning
    """

    tool_name = "reasoning"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)
        
        # 从环境变量加载API密钥
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        
        # 默认使用gpt-4o模型
        self.model = "gpt-4o"

        print("Intelligence Reasoning Service initialized")
        print(f"Using model: {self.model}")

    def _prepare_reasoning_prompt(self, question: str, original_task: str = "") -> str:
        """Prepare the reasoning prompt with question and optional context.

        Args:
            question: The main question for reasoning
            original_task: Optional original task description for context

        Returns:
            Formatted prompt string
        """
        if original_task:
            return f"Original Task: {original_task}\n\nQuestion: {question}"
        return f"Question: {question}"

    def _call_reasoning_model(self, prompt: str, temperature: float = 0.3) -> str:
        """Call the reasoning model with the prepared prompt.

        Args:
            prompt: The formatted prompt for reasoning
            temperature: Model temperature for response variability

        Returns:
            Reasoning result from the model

        Raises:
            Exception: If model call fails
        """
        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert at solving complex problems including math, "
                        "code contests, riddles, and puzzles. "
                        "Provide detailed step-by-step reasoning and a clear final answer."
                    ),
                },
                {"role": "user", "content": prompt},
        ]
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )

        # 返回模型响应内容
        return response.choices[0].message.content

    def mcp_complex_problem_reasoning(
        self,
        question: str = Field(
            description="The input question for complex problem reasoning, such as math and code contest problems"
        ),
        original_task: str = Field(default="", description="The original task description."),
        temperature: float = Field(
            default=0.3,
            description="Model temperature for response variability (0.0-1.0)",
            ge=0.0,
            le=1.0,
        ),
        reasoning_style: Literal["detailed", "concise", "step-by-step"] = Field(
            default="detailed",
            description="Style of reasoning output: detailed analysis, concise summary, or step-by-step breakdown",
        ),
    ) -> ActionResponse:
        """This tool provides comprehensive reasoning capabilities for:
        - Mathematical problems and proofs
        - Programming and algorithm challenges
        - Logic puzzles, brain teasers, and fun riddles
        - Competition-level STEM problems
        - Multi-step analytical reasoning tasks

        Weakness:
        - Inability to process media types: image, audio, or video
        - Require precise description of problem context and settings
        """
        try:
            # Handle FieldInfo objects
            if isinstance(question, FieldInfo):
                question = question.default
            if isinstance(original_task, FieldInfo):
                original_task = original_task.default
            if isinstance(temperature, FieldInfo):
                temperature = temperature.default
            if isinstance(reasoning_style, FieldInfo):
                reasoning_style = reasoning_style.default

            # Validate input
            if not question or not question.strip():
                raise ValueError("Question is required for complex problem reasoning")

            print(f"Processing reasoning request: {question[:100]}...")

            start_time = time.time()

            # Prepare the reasoning prompt
            prompt = self._prepare_reasoning_prompt(question, original_task)

            # Enhance prompt based on reasoning style
            if reasoning_style == "step-by-step":
                prompt += "\n\nPlease provide a clear step-by-step breakdown of your reasoning process."
            elif reasoning_style == "concise":
                prompt += "\n\nPlease provide a concise but complete reasoning and final answer."
            elif reasoning_style == "detailed":
                prompt += "\n\nPlease provide detailed analysis with comprehensive reasoning."

            # Call the reasoning model
            reasoning_result = self._call_reasoning_model(prompt, temperature)

            processing_time = time.time() - start_time

            # Prepare metadata
            metadata = {
                "model_name": self.model,
                "reasoning_style": reasoning_style,
                "response_length": len(reasoning_result),
            }

            print(
                f"Successfully completed reasoning ({len(reasoning_result)} characters, {processing_time:.2f}s)"
            )

            return ActionResponse(success=True, message=reasoning_result, metadata=metadata)

        except ValueError as e:
            self.logger.error(f"Invalid input: {str(e)}")
            return ActionResponse(
                success=False,
                message=f"Invalid input: {str(e)}",
                metadata={"error_type": "invalid_input", "error_message": str(e)},
            )
        except Exception as e:
            self.logger.error(f"Reasoning failed: {str(e)}: {traceback.format_exc()}")
            return ActionResponse(
                success=False,
                message=f"Reasoning failed: {str(e)}",
                metadata={"error_type": "reasoning_error", "error_message": str(e)},
            )

    def mcp_get_reasoning_capabilities(self) -> ActionResponse:
        """Get information about the reasoning service capabilities.

        Returns:
            ActionResponse with service capabilities and configuration
        """
        capabilities = {
            "Mathematical Problems": "Advanced mathematical reasoning, proofs, and calculations",
            "Code Contests": "Programming challenges, algorithm design, and optimization",
            "Logic Puzzles": "Brain teasers, riddles, and logical reasoning problems",
            "STEM Problems": "Competition-level science, technology, engineering, and math",
            "Multi-step Analysis": "Complex analytical reasoning with multiple interconnected steps",
        }

        capability_list = "\n".join(
            [f"**{capability}**: {description}" for capability, description in capabilities.items()]
        )

        metadata = {
            "model_name": self.model,
            "provider": "OpenAI",
            "supported_capabilities": list(capabilities.keys()),
            "total_capabilities": len(capabilities),
            "reasoning_styles": ["detailed", "concise", "step-by-step"],
        }

        return ActionResponse(
            success=True,
            message=f"Intelligence Reasoning Service Capabilities:\n\n{capability_list}",
            metadata=metadata,
        )


# Example usage and entry point
if __name__ == "__main__":
    import sys
    import json
    
    is_mcp_mode = len(sys.argv) == 1
    if is_mcp_mode:
        original_print = print
        print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
    
    load_dotenv()
    args = ActionArguments(
        name="intelligence_reasoning_service",
        transport="stdio",
    )
    
    try:
        service = ThinkCollection(args)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                function_name = input_data.get("function_name", input_data.get("name", ""))
                arguments = input_data.get("arguments", {})
                
                if function_name == "mcp_complex_problem_reasoning" or function_name == "reason":
                    result = service.mcp_complex_problem_reasoning(
                        question=arguments.get("question", ""),
                        original_task=arguments.get("original_task", ""),
                        temperature=arguments.get("temperature", 0.3),
                        reasoning_style=arguments.get("reasoning_style", "detailed")
                    )
                elif function_name == "mcp_get_reasoning_capabilities":
                    result = service.mcp_get_reasoning_capabilities()
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
