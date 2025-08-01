from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path

from mas_arena.tools.mcp_tool_transform import mcp_tool_desc_transform, call_mcp_tool

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages MCP tool descriptions and calls tools via a Sandbox."""
    def __init__(self, mcp_servers: Dict[str, Dict] = None, use_mcp_tools: bool = False, sandbox: Optional[Any] = None):
        self.mcp_servers = mcp_servers or {}
        self.sandbox = sandbox
        self.client = None
        self.tools: List[Any] = []
        self.use_mcp_tools = use_mcp_tools
        self._tool_descriptions: List[Any] = []
        logger.info(f"ToolManager initialized with {len(self.mcp_servers.get('mcpServers', {}))} MCP servers")

    async def setup(self):
        """Load tool descriptions."""
        if self.use_mcp_tools and self.mcp_servers:
            try:
                server_names = list(self.mcp_servers.get("mcpServers", {}).keys())
                logger.info(f"Loading tool descriptions for servers: {server_names}")
                
                tool_descriptions = await mcp_tool_desc_transform(
                    server_names,
                    {"mcpServers": self.mcp_servers.get("mcpServers", {})},
                    self.sandbox
                )
                
                self.tools.extend(tool_descriptions)
                self._tool_descriptions = tool_descriptions
                
                logger.info(f"Loaded {len(tool_descriptions)} MCP tools")
                tool_names = [tool.get('function_name', {}) for tool in tool_descriptions]
                logger.info(f"Loaded tools: {tool_names}")
                
            except Exception as e:
                logger.error(f"Error preparing MCP tools: {e}", exc_info=True)
        return self

    def get_tools(self) -> List[Any]:
        """Get the list of all loaded tools."""
        return self._tool_descriptions
        
    async def call_tool(self, server_name: str, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Call MCP tool via the sandbox."""
        if not self.sandbox:
            return {"error": "Sandbox not configured in ToolManager"}

        try:
            logger.info(f"Calling tool {function_name} on server {server_name}")
            
            config = {"mcpServers": self.mcp_servers.get("mcpServers", {})}
            
            result = await call_mcp_tool(
                server_name=server_name,
                function_name=function_name,
                parameters=parameters,
                mcp_config=config,
                sandbox=self.sandbox
            )
            
            if "error" in result:
                logger.warning(f"Tool call error: {result['error']}")
            else:
                logger.info(f"Tool call successful: {server_name}.{function_name}")
                
            return result
        except Exception as e:
            logger.error(f"Error calling tool {function_name} on {server_name}: {e}", exc_info=True)
            return {"error": str(e)}

    @classmethod
    def from_config_file(cls, config_file_path: str, sandbox: Any) -> "ToolManager":
        """Create a ToolManager instance from a configuration file."""
        try:
            config_path = Path(config_file_path)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file_path}")
                return cls({}, use_mcp_tools=True, sandbox=sandbox)
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            mcp_servers = config.get("mcpServers", {})
            return cls(mcp_servers={"mcpServers": mcp_servers}, use_mcp_tools=True, sandbox=sandbox)
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}", exc_info=True)
            return cls({}, use_mcp_tools=True, sandbox=sandbox)
