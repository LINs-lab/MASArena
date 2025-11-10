"""
This module initializes the agent systems available in the MAS Arena.

It dynamically imports all agent system modules in the current directory,
registers them with the AgentSystemRegistry, and makes them available
for creation through the `create_agent_system` function.
"""
import pkgutil
import importlib
import traceback

# Import the base classes and registry
from .base import AgentSystem, AgentSystemRegistry, create_agent_system

# --- Dynamic Discovery and Registration ---
# Iterate over all modules in the current package path
# and import them. This is what triggers the registration decorators
# in each agent system file to run and register themselves.
for _, name, _ in pkgutil.iter_modules(__path__):
    # Ensure we don't try to import the base module itself again
    # or any other non-agent system modules.
    if name not in ['base', 'format_prompts']:
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
]
