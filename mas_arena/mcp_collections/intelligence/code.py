import os
import time
import traceback
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


from ..base import ActionArguments, ActionCollection, ActionResponse

class CodeGenerationMetadata(BaseModel):
    """Metadata for code generation results."""

    model_name: str | None = None
    code_style: str | None = None
    code_length: int | None = None
    line_count: int | None = None
    processing_time_seconds: float | None = None
    temperature: float | None = None
    has_requirements: bool | None = None
    has_context: bool | None = None
    saved_file_path: str | None = None
    file_save_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class CodeCollection(ActionCollection):
    """MCP service for generating executable Python code snippets using LLM.

    Supports code generation for:
    - Data processing and analysis tasks
    - Algorithm implementations
    - Utility functions and scripts
    - Problem-solving code snippets
    - Educational programming examples
    """
    tool_name = "code"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)


    def _prepare_code_prompt(self, task_description: str, requirements: str = "", context: str = "") -> str:
        """Prepare the code generation prompt with task description and optional requirements.

        Args:
            task_description: The main task for code generation
            requirements: Optional specific requirements or constraints
            context: Optional additional context or background information

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Task: {task_description}"]

        if requirements:
            prompt_parts.append(f"Requirements: {requirements}")

        if context:
            prompt_parts.append(f"Context: {context}")

        return "\n\n".join(prompt_parts)

    def _call_code_model(self, prompt: str, temperature: float = 0.1) -> str:
        """Call the code generation model with the prepared prompt.

        Args:
            prompt: The formatted prompt for code generation
            temperature: Model temperature for response variability

        Returns:
            Generated code from the model

        Raises:
            Exception: If model call fails
        """
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_API_URL"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python programmer. Generate clean, efficient, and "
                        "well-documented Python code that solves the given task. "
                        "Include proper error handling and follow Python best practices. "
                        "Return only executable Python code with minimal explanatory comments."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature
        )

        return response.content

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from the model response.

        Args:
            response: Raw response from the model

        Returns:
            Extracted Python code
        """
        # Remove markdown code blocks if present
        lines = response.strip().split("\n")

        # Find code block boundaries
        start_idx = 0
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if line.strip().startswith("```python") or line.strip().startswith("```"):
                start_idx = i + 1
                break

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break

        # Extract the code
        code_lines = lines[start_idx:end_idx]
        return "\n".join(code_lines).strip()

    def mcp_generate_python_code(
        self,
        task_description: str = Field(description="Description of the programming task or problem to solve"),
        requirements: str = Field(
            default="", description="Specific requirements, constraints, or specifications for the code"
        ),
        context: str = Field(default="", description="Additional context or background information"),
        temperature: float = Field(
            default=0.1,
            description="Model temperature for code generation (0.0-1.0, lower = more deterministic)",
            ge=0.0,
            le=1.0,
        ),
        code_style: Literal["minimal", "documented", "verbose"] = Field(
            default="documented",
            description="Style of generated code: minimal (concise), documented (with comments), verbose (detailed)",
        ),
        save_to_file_path: str | None = Field(
            default=None,
            description="Optional. Path to save the generated Python snippet. e.g., 'output/generated_script.py'",
        ),
    ) -> ActionResponse:
        """Generate executable Python code snippets based on task description.

        This tool provides comprehensive code generation capabilities for:
        - Solve simple math tasks and validations
        - Data processing and analysis tasks
        - Algorithm implementations and optimizations
        - Utility functions and helper scripts
        - Problem-solving code snippets
        - Educational programming examples
        - API integrations and automation scripts
        """
        try:
            # Handle FieldInfo objects
            if isinstance(task_description, FieldInfo):
                task_description = task_description.default
            if isinstance(requirements, FieldInfo):
                requirements = requirements.default
            if isinstance(context, FieldInfo):
                context = context.default
            if isinstance(temperature, FieldInfo):
                temperature = temperature.default
            if isinstance(code_style, FieldInfo):
                code_style = code_style.default
            if isinstance(save_to_file_path, FieldInfo):
                save_to_file_path = save_to_file_path.default

            # Validate input
            if not task_description or not task_description.strip():
                raise ValueError("Task description is required for code generation")

            print(f"Generating code for: {task_description[:100]}...")

            start_time = time.time()

            # Prepare the code generation prompt
            prompt = self._prepare_code_prompt(task_description, requirements, context)

            # Enhance prompt based on code style
            if code_style == "minimal":
                prompt += "\n\nGenerate concise, minimal code without extensive comments."
            elif code_style == "verbose":
                prompt += "\n\nGenerate detailed code with comprehensive comments and explanations."
            elif code_style == "documented":
                prompt += "\n\nGenerate well-documented code with clear comments and docstrings."

            # Call the code generation model
            raw_response = self._call_code_model(prompt, temperature)

            # Extract clean Python code
            generated_code = self._extract_python_code(raw_response)

            processing_time = time.time() - start_time

            # Populate metadata fields
            metadata = CodeGenerationMetadata(
                model_name=self._llm_config.llm_model_name,
                code_style=code_style,
                code_length=len(generated_code),
                line_count=len(generated_code.split("\n")),
                processing_time_seconds=round(processing_time, 2),
                temperature=temperature,
                has_requirements=bool(requirements.strip()),
                has_context=bool(context.strip()),
            )

            # Save the generated code to a file if path is provided
            if save_to_file_path:
                try:
                    # Use _validate_file_path to ensure path is within workspace and get absolute path
                    # The check_existence=False allows creating a new file.
                    output_file_path_obj = Path(self._validate_file_path(save_to_file_path))

                    # Ensure parent directories exist
                    output_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_file_path_obj, "w", encoding="utf-8") as f:
                        f.write(generated_code)

                    metadata.saved_file_path = str(output_file_path_obj)
                    print(f"Generated code also saved to: {output_file_path_obj}")
                except Exception as e:
                    self.logger.error(f"Failed to save code to file '{save_to_file_path}': {str(e)}")
                    metadata.file_save_error = str(e)

            self._color_log(
                f"Successfully generated code ({metadata.code_length} characters, "
                f"{metadata.processing_time_seconds:.2f}s)",
                Color.green,
            )

            return ActionResponse(success=True, message=generated_code, metadata=metadata.model_dump(exclude_none=True))

        except ValueError as e:
            self.logger.error(f"Invalid input: {str(e)}")
            metadata.error_type = "invalid_input"
            metadata.error_message = str(e)
            return ActionResponse(
                success=False,
                message=f"Invalid input: {str(e)}",
                metadata=metadata.model_dump(exclude_none=True),
            )
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}: {traceback.format_exc()}")
            metadata.error_type = "generation_error"
            metadata.error_message = str(e)
            return ActionResponse(
                success=False,
                message=f"Code generation failed: {str(e)}",
                metadata=metadata.model_dump(exclude_none=True),
            )

    def mcp_get_code_capabilities(self) -> ActionResponse:
        """Get information about the code generation service capabilities.

        Returns:
            ActionResponse with service capabilities and configuration
        """
        capabilities = {
            "Data Processing": "Generate code for data manipulation, analysis, and visualization",
            "Algorithm Implementation": "Create efficient algorithms and data structures",
            "Utility Functions": "Build helper functions and reusable code components",
            "Problem Solving": "Generate solutions for programming challenges and tasks",
            "API Integration": "Create code for working with APIs and web services",
            "Automation Scripts": "Build scripts for task automation and workflow optimization",
        }

        capability_list = "\n".join(
            [f"**{capability}**: {description}" for capability, description in capabilities.items()]
        )

        metadata = {
            "model_name": self._llm_config.llm_model_name,
            "provider": self._llm_config.llm_provider,
            "supported_capabilities": list(capabilities.keys()),
            "total_capabilities": len(capabilities),
            "code_styles": ["minimal", "documented", "verbose"],
            "python_version": ">=3.11",
            "supported_language": "Python",
        }

        return ActionResponse(
            success=True,
            message=f"Code Generation Service Capabilities:\n\n{capability_list}",
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
        name="code_generation_service",
        transport="stdio",
    )
    
    try:
        service = CodeCollection(args)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                function_name = input_data.get("function_name", input_data.get("name", ""))
                arguments = input_data.get("arguments", {})
                
                if function_name == "mcp_generate_python_code":
                    result = service.mcp_generate_python_code(
                        task_description=arguments.get("task_description", ""),
                        requirements=arguments.get("requirements", ""),
                        context=arguments.get("context", ""),
                        temperature=arguments.get("temperature", 0.1),
                        code_style=arguments.get("code_style", "documented"),
                        save_to_file_path=arguments.get("save_to_file_path", None)
                    )
                elif function_name == "mcp_get_code_capabilities":
                    result = service.mcp_get_code_capabilities()
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
