from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
import logging

from mas_arena.tools.base import ToolFactory

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages MCP tool servers and provides tools to agent systems."""
    def __init__(self, mcp_servers: Dict[str, Dict] = None, mock_mode: bool = False, tool_assignment_rules: Optional[Dict[str, List[str]]] = None, use_local_tools: bool = False, use_mcp_tools: bool = False):
        self.mcp_servers = mcp_servers or {}
        self.client = None
        self.tools: List[Any] = []
        self.mock_mode = mock_mode
        self.use_local_tools = use_local_tools
        self.use_mcp_tools = use_mcp_tools
        # Optional mapping of agent names to lists of tool names (assignment rules)
        self.tool_assignment_rules: Dict[str, List[str]] = tool_assignment_rules or {}
        self._exit_stack = AsyncExitStack()
        
        # Load local tools if requested
        if self.use_local_tools:
            self.load_local_tools()

        # Mock tools are loaded if mock_mode is on, regardless of other settings
        if self.mock_mode:
            self.tools.extend(self._create_mock_tools())

    def load_local_tools(self):
        """
        Load tools registered with the ToolFactory.
        This method instantiates tool providers and stores them along with their registration metadata.
        The actual tool objects are retrieved and formatted later by `_convert_tool_format`.
        """
        logger.info("Loading local tool providers registered with ToolFactory...")
        for name, tool_cls in ToolFactory._cls.items():
            if name.startswith("async_"):
                continue

            try:
                provider_instance = tool_cls()
                # Store the instance along with its registration name to fetch category later
                self.tools.append({"provider": provider_instance, "name": name, "source": "local"})
                logger.info(f"Successfully loaded local tool provider: {name}")
            except Exception as e:
                logger.error(f"Failed to instantiate tool provider '{name}': {e}", exc_info=True)

    async def __aenter__(self):
        # If not in mock mode and MCP tools are requested, connect to servers
        if not self.mock_mode and self.use_mcp_tools and self.mcp_servers:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                
                self.client = MultiServerMCPClient(self.mcp_servers)
                await self._exit_stack.enter_async_context(self.client)
                # Append remote tools to the existing tool list
                self.tools.extend(await self.client.get_tools())

            except ImportError as e:
                print(f"Error: langchain_mcp_adapters not found - {e}. MCP tools disabled.")
            except Exception as e:
                print(f"Error connecting to MCP servers: {e}. MCP tools disabled.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def _create_mock_tools(self) -> List[Any]:
        """Create mock tools for testing."""
        # ... (original mock tool creation logic)
        return []

    def get_tools(self) -> List[Any]:
        """Get the list of all loaded tools in a unified format for ToolSelector."""
        return self._convert_tool_format(self.tools)
    
    def _convert_tool_format(self, tools_input: List[Any]) -> List[Dict[str, Any]]:
        """
        Converts a list of mixed tool providers and pre-formatted tool dictionaries
        into a unified list of tool dictionaries for the ToolSelector.
        """
        final_tools = []
        for item in tools_input:
            if isinstance(item, dict) and "source" in item and item["source"] == "local":
                # This is a local tool provider that needs processing
                provider = item.get("provider")
                name = item.get("name")
                if not provider or not name or not hasattr(provider, 'get_tools'):
                    continue
                
                try:
                    ext_info = ToolFactory.get_ext_info(name)
                    category = ext_info.get("category", "general")
                    
                    actual_tools = provider.get_tools()
                    for tool in actual_tools:
                        final_tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "category": category,
                            "tool_object": tool
                        })
                except Exception as e:
                    logger.error(f"Failed to process tools from provider '{name}': {e}", exc_info=True)

            elif isinstance(item, dict):
                # Assume it's an already formatted tool dictionary (e.g., from MCP)
                final_tools.append(item)
            else:
                # Handle other potential tool formats if necessary
                logger.warning(f"Item in tool list is not a dictionary or a recognized format: {item}")
        
        return final_tools
        
    def get_tool_assignment_rules(self) -> Dict[str, List[str]]:
        """Return the tool assignment rules loaded from configuration."""
        return self.tool_assignment_rules

    @classmethod
    def from_config_file(cls, config_file_path: str, mock_mode: bool = False) -> "ToolManager":
        # ... (original class method logic)
            return cls({}, mock_mode=True, tool_assignment_rules={}) 