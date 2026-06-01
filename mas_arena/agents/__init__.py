"""
This module initializes the agent systems available in the MAS Arena.

It dynamically imports all agent system modules in the current directory,
registers them with the AgentSystemRegistry, and makes them available
for creation through the `create_agent_system` function.
"""
import pkgutil
import importlib
import os
import traceback

# Import the base classes and registry
from .base import AgentSystem, AgentSystemRegistry, create_agent_system

LEGACY_AGENT_MODULES = {
    "autogen",
    "camel",
    "chateval",
    "evoagent",
    "jarvis",
    "metagpt",
    "swarm",
}

# --- Dynamic Discovery and Registration ---
# Iterate over all modules in the current package path
# and import them. This is what triggers the registration decorators
# in each agent system file to run and register themselves.
for _, name, _ in pkgutil.iter_modules(__path__):
    # Ensure we don't try to import the base module itself again
    # or any other non-agent system modules.
    skip_modules = {'base', 'format_prompts', 'agent_core'}
    if os.getenv("MAS_ARENA_ENABLE_LEGACY_AGENTS") != "true":
        skip_modules.update(LEGACY_AGENT_MODULES)
    if name not in skip_modules:
            importlib.import_module(f".{name}", __package__)

# --- Public API ---
# Expose the populated registry and convenient dictionaries for the application.

# Get all registered agent systems
AVAILABLE_AGENT_SYSTEMS = AgentSystemRegistry.get_all_systems()

# Define what gets imported when a user does 'from mas_arena.agents import *'
__all__ = [
    "AgentSystem",
    "AgentSystemRegistry",
    "create_agent_system",
    "AVAILABLE_AGENT_SYSTEMS",
    "BenchAgent",  # 新增BenchAgent类
]

# 导入BenchAgent以便直接使用
try:
    from .bench_agent import BenchAgent
except ImportError:
    pass  # 如果导入失败，忽略错误
