from typing import Any, List, Dict, Optional
import random
from collections import defaultdict

class ToolSelector:
    """
    Primary interface for tool selection across single-agent and multi-agent scenarios.
    Use `select_tools(task_description, num_agents=None, overlap=False, limit=5)` to pick or partition tools.
    Override `select_tools` to implement custom selection strategies.
    """

    def __init__(self, tools: List[Dict[str, Any]]):
        """
        Initialize with available tools.
        
        Args:
            tools: List of tool definitions from ToolManager
        """
        self.tools = tools
        # Index tools by category for faster lookup
        self.tools_by_category = defaultdict(list)
        for tool in tools:
            category = tool.get("category", "general")
            self.tools_by_category[category].append(tool)
    
    def _select_for_task(self, task_description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Internal single-agent tool selection logic.
        """
        # In a production system, this would use embeddings or LLM-based selection
        # For now, we'll use simple keyword matching
        
        # Convert to lowercase for case-insensitive matching
        task_lower = task_description.lower()
        
        # Define some basic heuristics
        tool_scores = []
        
        for tool in self.tools:
            score = 0
            # Check if tool name or description appears in task
            name = tool.get("name", "").lower()
            description = tool.get("description", "").lower()
            
            # Direct mentions get higher scores
            if name in task_lower:
                score += 5
            
            # Check keywords in description
            for keyword in description.split():
                if keyword and len(keyword) > 3 and keyword in task_lower:
                    score += 1
                    
            # Check for domain-specific keywords
            if "math" in task_lower and tool.get("category") == "math":
                score += 3
            if "search" in task_lower and (tool.get("category") == "search" or "search" in name):
                score += 3
            if ("code" in task_lower or "programming" in task_lower) and tool.get("category") == "code":
                score += 3
                
            tool_scores.append((tool, score))
        
        # Sort by score (descending) and take top k
        selected = [t for t, s in sorted(tool_scores, key=lambda x: x[1], reverse=True)[:limit]]
        
        # If nothing was selected with a good score, include general tools
        if not selected:
            selected = self.tools_by_category.get("general", [])[:limit]
            
        return selected
    
    def _partition_tools_for_multi_agent(
        self,
        num_agents: int,
        overlap: bool = False,
        task_description: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Internal multi-agent tool partitioning logic.
        """
        if num_agents <= 0:
            return []
            
        # If only one agent, give it all tools
        if num_agents == 1:
            return [self.tools]
            
        # If task description provided, use relevance-based partitioning
        if task_description:
            # Select top tools for the task
            relevant_tools = self._select_for_task(task_description, limit=len(self.tools))
            
            # Create partitions
            partitions = [[] for _ in range(num_agents)]
            
            # Assign tools round-robin
            for i, tool in enumerate(relevant_tools):
                agent_idx = i % num_agents
                partitions[agent_idx].append(tool)
                
            return partitions
        
        # Otherwise, partition by category
        partitions = [[] for _ in range(num_agents)]
        
        # First, distribute categories to agents
        categories = list(self.tools_by_category.keys())
        for i, category in enumerate(categories):
            agent_idx = i % num_agents
            tools_in_category = self.tools_by_category[category]
            
            if overlap:
                # Add all tools in category to this agent
                partitions[agent_idx].extend(tools_in_category)
            else:
                # Split category tools among agents
                for j, tool in enumerate(tools_in_category):
                    if overlap:
                        # Add to all agents
                        for p in partitions:
                            p.append(tool)
                    else:
                        # Add to one agent
                        sub_idx = (agent_idx + j) % num_agents
                        partitions[sub_idx].append(tool)
        
        # Ensure each agent has at least one tool
        for i, partition in enumerate(partitions):
            if not partition:
                # If this agent has no tools, give it a random one
                random_tool = random.choice(self.tools)
                partitions[i].append(random_tool)
                
        return partitions

    def select_by_names(self, tool_names: List[str]) -> List[Any]:
        """Select tools by their names (case-insensitive)."""
        name_set = set(n.lower() for n in tool_names)
        return [tool for tool in self.tools if getattr(tool, 'name', '').lower() in name_set]

    def filter_by_roles(self, role_patterns: Dict[str, List[str]]) -> Dict[str, List[Any]]:
        """Filter tools for each role based on name patterns."""
        role_tools: Dict[str, List[Any]] = {}
        for role, patterns in role_patterns.items():
            selected = []
            for tool in self.tools:
                name = getattr(tool, 'name', '').lower()
                if any(pat.lower() in name for pat in patterns):
                    selected.append(tool)
            role_tools[role] = selected
        return role_tools
        
    def filter_by_keywords(self, keywords: List[str], match_all: bool = False) -> List[Any]:
        """
        Filter tools that match specified keywords in name or description.
        
        Args:
            keywords: List of keywords to match
            match_all: If True, tools must match all keywords. If False, match any keyword.
        
        Returns:
            List of tools matching the criteria
        """
        results = []
        for tool in self.tools:
            name = getattr(tool, 'name', '').lower()
            desc = getattr(tool, 'description', '').lower()
            
            matches = [kw.lower() in name or kw.lower() in desc for kw in keywords]
            
            if match_all and all(matches) or not match_all and any(matches):
                results.append(tool)
                
        return results

    def select_tools(
        self,
        task_description: str,
        num_agents: Optional[int] = None,
        overlap: bool = False,
        limit: int = 5
    ) -> Any:
        """
        Unified public interface for tool selection.
        - Single-agent (num_agents None or <=1): returns a flat list of top tools.
        - Multi-agent (num_agents >1): returns a list of tool lists, one per agent.
        Override this method to implement any custom logic.
        """
        if num_agents and num_agents > 1:
            return self._partition_tools_for_multi_agent(num_agents, overlap, task_description)
        return self._select_for_task(task_description, limit)

