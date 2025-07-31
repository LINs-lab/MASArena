import logging
from typing import Dict, Any, Optional

from mas_arena.tools.tool_selector import ToolSelector
from mas_arena.tools.tool_manager import ToolManager
from mas_arena.agents.base import AgentSystem

# Set up a logger for tool integration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

class ToolIntegrationWrapper(AgentSystem):
    """
    Wraps any AgentSystem to inject LLM-driven tool selection.
    """
    def __init__(self, inner: AgentSystem, config: Dict[str, Any], tool_manager: Optional[ToolManager] = None):
        """Initialize with an inner agent system and its configuration."""
        super().__init__(name=f"{inner.name}_with_tools", config=config)
        self.inner = inner
        self.tool_manager = tool_manager
        self.tool_selector: Optional[ToolSelector] = None

    async def setup(self):
        """
        Set up the tool manager and selector, then patch the inner agent system.
        """
        use_mcp_tools = self.config.get("use_mcp_tools", False)
        
        if use_mcp_tools:
            if not self.tool_manager:
                logger.error(f"ToolIntegrationWrapper requires a pre-initialized ToolManager instance.")
                return

            # The inner agent must have an LLM client to pass to the selector
            if not hasattr(self.inner, 'client') or self.inner.client is None:
                logger.error(f"Agent '{self.inner.name}' must have a 'client' attribute for LLM-based tool selection.")
                return

            all_tools = self.tool_manager.get_tools()
            
            # Pass the agent's LLM client to the selector
            self.tool_selector = ToolSelector(all_tools, self.inner.client)
            
            logger.info(f"Initialized ToolSelector with {len(all_tools)} tools for LLM-based selection.")
            self._apply_patches()
        else:
            logger.info("`use_mcp_tools` is false, skipping tool integration setup.")

        if hasattr(self.inner, 'setup'):
            await self.inner.setup()

    async def select_tools_for_problem(self, problem: Any, num_agents: Optional[int] = None) -> Any:
        """
        Select or partition tools for a given problem using the LLM-based selector.
        """
        if not self.tool_selector:
            logger.warning("ToolSelector not initialized. Returning no tools.")
            return [] if num_agents is None or num_agents <= 1 else [[] for _ in range(num_agents or 0)]

        problem_desc = problem.get("problem", "") if isinstance(problem, dict) else str(problem)
        return await self.tool_selector.select_tools(
            problem_desc,
            num_agents=num_agents,
            limit=self.config.get("tool_limit", 10) # Use a sensible default limit
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Delegate to inner agent's run_agent method."""
        # This run_agent will be the one on the *wrapped* instance.
        # The patching logic replaces the *inner* agent's run_agent.
        return await self.inner.run_agent(problem, **kwargs)

    def _apply_patches(self):
        """Apply the appropriate method patches based on agent system type."""
        # For now, we only focus on the single-agent case as it's our primary use case
        if isinstance(self.inner, AgentSystem) and not hasattr(self.inner, "_create_agents"):
             self._patch_single_agent_system()
        else:
            logger.warning(f"Tool integration patching not implemented for agent type: {self.inner.__class__.__name__}")

    def _patch_single_agent_system(self):
        """Patch a single-agent system to select tools before running."""
        original_run_agent = self.inner.run_agent
        
        async def patched_run_agent(problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            logger.info(f"[{self.name}] Selecting tools for problem via LLM...")
            
            # Asynchronously select a subset of tools for the current task
            selected_tools = await self.select_tools_for_problem(problem)
            logger.info(f"LLM selected {len(selected_tools)} tools: {[t.get('function_name', t.get('name')) for t in selected_tools]}")

            # Backup the agent's original full tool list (if any)
            original_tools = getattr(self.inner, 'tools', [])
            # Temporarily assign the *selected* tools to the agent for this run
            self.inner.tools = selected_tools
            self.inner.tool_manager = self.tool_manager
            
            try:
                # Run the agent's original logic with the dynamically selected tools
                result = await original_run_agent(problem, **kwargs)
            finally:
                # IMPORTANT: Restore the original full tool list after the run
                self.inner.tools = original_tools
                
            return result

        # Apply the patch
        self.inner.run_agent = patched_run_agent
        logger.info(f"Successfully patched `run_agent` on '{self.inner.name}' for LLM-driven tool selection.")

    def __getattr__(self, name):
        """Delegate all other attribute access to inner agent system."""
        return getattr(self.inner, name)

    async def teardown(self):
        """Clean up the inner agent's resources."""
        # The shared tool_manager is cleaned up by the BenchmarkRunner.
        if hasattr(self.inner, 'teardown'):
            await self.inner.teardown()
