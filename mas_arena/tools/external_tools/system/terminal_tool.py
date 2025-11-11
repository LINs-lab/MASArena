from typing import Optional, Dict, Any
import json
import time
import os
import platform
import subprocess
import asyncio
from datetime import datetime

from smolagents import Tool
from smolagents.models import MessageRole, Model

from dotenv import load_dotenv

load_dotenv()


class TerminalTool(Tool):
    name = "execute_terminal_command"
    description = """
Execute terminal/shell commands safely with timeout controls.
This tool supports cross-platform command execution (Windows, macOS, Linux) and provides 
safety checks for dangerous commands. Useful for running system commands, scripts, and programs.
"""

    inputs = {
        "command": {
            "description": "Terminal command to execute",
            "type": "string",
        },
        "timeout": {
            "description": "Command timeout in seconds (default: 30, max: 300)",
            "type": "integer",
            "nullable": True,
        },
        "working_directory": {
            "description": "Working directory to execute the command in (optional)",
            "type": "string",
            "nullable": True,
        },
        "capture_output": {
            "description": "Whether to capture and return command output (default: true)",
            "type": "boolean",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int = 8000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

        # Initialize command history
        self.command_history: list[dict] = []
        self.max_history_size = 50

        # Define dangerous commands for safety
        self.dangerous_commands = [
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",  # Unix fork bomb
            "del /f /s /q",
            "format",
            "diskpart",  # Windows
            "sudo rm",
            "sudo dd",
            "sudo mkfs",  # Sudo variants
        ]

        # Get current platform info
        self.platform_info = {
            "system": platform.system(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        }

    def _check_command_safety(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if command is safe to execute."""
        command_lower = command.lower().strip()

        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd.lower() in command_lower:
                return False, f"Dangerous command detected: '{dangerous_cmd}'"

        return True, None

    def _execute_command(
        self,
        command: str,
        timeout: int = 30,
        working_directory: Optional[str] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """Execute command synchronously with timeout."""
        start_time = datetime.now()

        try:
            # Safety check
            is_safe, reason = self._check_command_safety(command)
            if not is_safe:
                return {
                    "command": command,
                    "success": False,
                    "stdout": "",
                    "stderr": reason,
                    "return_code": -1,
                    "duration": "0.00s",
                    "timestamp": start_time.isoformat(),
                    "safety_check_passed": False,
                }

            # Prepare command for different platforms
            if self.platform_info["system"] == "Windows":
                cmd_args = ["cmd", "/c", command]
            else:
                cmd_args = ["/bin/bash", "-c", command]

            # Execute command
            if capture_output:
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=working_directory,
                )
                stdout = result.stdout
                stderr = result.stderr
            else:
                result = subprocess.run(
                    cmd_args, timeout=timeout, cwd=working_directory
                )
                stdout = "[Output not captured]"
                stderr = ""

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = result.returncode == 0

            # Store in history
            history_entry = {
                "command": command,
                "success": success,
                "return_code": result.returncode,
                "duration": f"{duration:.2f}s",
                "timestamp": start_time.isoformat(),
                "working_directory": working_directory or os.getcwd(),
            }

            self.command_history.append(history_entry)
            if len(self.command_history) > self.max_history_size:
                self.command_history.pop(0)

            return {
                "command": command,
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": result.returncode,
                "duration": f"{duration:.2f}s",
                "timestamp": start_time.isoformat(),
                "safety_check_passed": True,
            }

        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            return {
                "command": command,
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "return_code": -1,
                "duration": f"{duration:.2f}s",
                "timestamp": start_time.isoformat(),
                "safety_check_passed": True,
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return {
                "command": command,
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "return_code": -1,
                "duration": f"{duration:.2f}s",
                "timestamp": start_time.isoformat(),
                "safety_check_passed": True,
            }

    def _format_command_output(self, result: Dict[str, Any]) -> str:
        """Format command execution results for LLM consumption."""
        output_parts = [
            f"# Terminal Command Execution",
            "",
            f"**Command:** `{result['command']}`",
            f"**Status:** {'✅ SUCCESS' if result['success'] else '❌ FAILED'}",
            f"**Duration:** {result['duration']}",
            f"**Return Code:** {result['return_code']}",
            f"**Platform:** {self.platform_info['system']}",
        ]

        if not result["safety_check_passed"]:
            output_parts.extend(
                ["", "⚠️ **SAFETY CHECK FAILED**", f"**Reason:** {result['stderr']}"]
            )
            return "\n".join(output_parts)

        if result["stdout"]:
            # Truncate stdout if too long
            stdout = result["stdout"]
            if len(stdout) > self.text_limit:
                stdout = stdout[: self.text_limit] + "\n\n... [Output truncated]"

            output_parts.extend(["", "## Standard Output:", "```", stdout, "```"])

        if result["stderr"]:
            # Truncate stderr if too long
            stderr = result["stderr"]
            if len(stderr) > 1000:  # Shorter limit for errors
                stderr = stderr[:1000] + "\n\n... [Error output truncated]"

            output_parts.extend(["", "## Standard Error:", "```", stderr, "```"])

        return "\n".join(output_parts)

    def _get_command_history(self, count: int = 10) -> str:
        """Get recent command execution history."""
        if not self.command_history:
            return "No command history available."

        recent_commands = (
            self.command_history[-count:] if count > 0 else self.command_history
        )

        output_parts = [
            f"# Command History",
            f"\nShowing last {len(recent_commands)} commands:",
            "",
        ]

        for i, cmd in enumerate(recent_commands, 1):
            status = "✅" if cmd["success"] else "❌"
            output_parts.extend(
                [
                    f"## {i}. {status} `{cmd['command']}`",
                    f"**Time:** {cmd['timestamp'][:19]}",
                    f"**Duration:** {cmd['duration']}",
                    f"**Return Code:** {cmd['return_code']}",
                    f"**Directory:** {cmd['working_directory']}",
                    "",
                ]
            )

        return "\n".join(output_parts)

    def forward(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_directory: Optional[str] = None,
        capture_output: Optional[bool] = None,
    ) -> str:

        # Set defaults
        timeout = min(timeout or 30, 300)  # Max 5 minutes
        capture_output = capture_output if capture_output is not None else True

        # Special command for getting history
        if command.lower() in ["history", "get_history", "show_history"]:
            return self._get_command_history()

        # Execute the command
        result = self._execute_command(
            command, timeout, working_directory, capture_output
        )

        # Format and return result
        return self._format_command_output(result)
