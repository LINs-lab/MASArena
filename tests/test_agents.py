"""Tests for agent systems."""

import pytest
from unittest.mock import Mock, patch

from mas_arena.agents import (
    AgentSystemRegistry,
    create_agent_system,
    AVAILABLE_AGENT_SYSTEMS
)
from mas_arena.agents.base import AgentSystem


class TestAgentSystemRegistry:
    """Test agent system registry functionality."""
    
    def test_registry_has_systems(self):
        """Test that the registry contains registered systems."""
        systems = AgentSystemRegistry.get_all_systems()
        assert isinstance(systems, dict)
        assert len(systems) > 0
    
    def test_available_agent_systems_populated(self):
        """Test that AVAILABLE_AGENT_SYSTEMS is populated."""
        assert isinstance(AVAILABLE_AGENT_SYSTEMS, dict)
        assert len(AVAILABLE_AGENT_SYSTEMS) > 0
    
    def test_single_agent_exists(self):
        """Test that single_agent system is registered."""
        assert "single_agent" in AVAILABLE_AGENT_SYSTEMS


class TestAgentCreation:
    """Test agent system creation."""
    
    def test_create_single_agent(self, sample_agent_config):
        """Test creating a single agent system."""
        agent = create_agent_system("single_agent", sample_agent_config)
        assert agent is not None
        assert agent.name == "single_agent"
        assert hasattr(agent, "evaluate")    
    def test_create_invalid_agent(self, sample_agent_config):
        """Test creating an invalid agent system returns None."""
        agent = create_agent_system("invalid_agent", sample_agent_config)
        assert agent is None
    
    def test_create_agent_with_empty_config(self):
        """Test creating agent with empty configuration raises ValueError."""
        with pytest.raises(ValueError):
            create_agent_system("single_agent", {})


class TestAgentSystemBase:
    """Test base agent system functionality."""
    
    def test_agent_evaluate_interface(self, sample_problem, sample_agent_config):
        """Test that agent systems have evaluate interface."""
        # Create a mock agent for testing
        agent = create_agent_system("single_agent", sample_agent_config)
        assert hasattr(agent, "evaluate")
        
        # Test that evaluate method exists and is callable
        assert callable(getattr(agent, "evaluate", None))