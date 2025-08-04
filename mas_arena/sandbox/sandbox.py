import asyncio
import subprocess
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Sandbox:
    """Manages isolated environments for running agent tools."""
    def __init__(self, mcp_config: Dict[str, Any], workspace_dir: Optional[str] = None):
        self._server_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._mcp_config = mcp_config
        self.workspace_dir = workspace_dir

    async def get_server_process(self, server_name: str) -> Optional[asyncio.subprocess.Process]:
        """
        Starts and returns a new process for a given MCP server.
        Each call creates a fresh process since MCP servers in stdio mode 
        are designed to handle one request and exit.
        """
        server_config = self._mcp_config.get("mcpServers", {}).get(server_name)
        if not server_config:
            logger.error(f"Server '{server_name}' not found in MCP config.")
            return None
        
        server_type = server_config.get("type", "stdio")
        if server_type != "stdio":
            logger.info(f"Server '{server_name}' is of type '{server_type}', not a managed process. Skipping.")
            return None

        command = server_config.get("command")
        args = server_config.get("args", [])

        # For python commands, add -u flag for unbuffered output
        if command == "python":
            command = "python -u"
        
        full_command = f"{command} {' '.join(args)}".strip()

        if not command:
            logger.error(f"No command found for server '{server_name}' in MCP config.")
            return None

        try:
            # Add workspace to the command if the placeholder is present
            if self.workspace_dir and "{workspace_dir}" in full_command:
                command = full_command.format(workspace_dir=self.workspace_dir)
            else:
                command = full_command

            logger.info(f"Starting MCP server '{server_name}' with command: {command}")
            process = await asyncio.create_subprocess_shell( command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            return process
        except Exception as e:
            logger.error(f"Failed to start server '{server_name}': {e}", exc_info=True)
            return None

    async def cleanup_all_processes(self):
        """Terminates all running MCP server processes."""
        logger.info(f"Cleaning up {len(self._server_processes)} MCP processes.")
        for name, proc in self._server_processes.items():
            try:
                logger.info(f"Terminating MCP process '{name}' (PID: {proc.pid})...")
                proc.terminate()
                await proc.wait() 
                logger.info(f"Process '{name}' terminated.")
            except ProcessLookupError:
                logger.warning(f"Process '{name}' (PID: {proc.pid}) was not found. It might have already terminated.")
        self._server_processes.clear()
