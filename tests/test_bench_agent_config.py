from types import SimpleNamespace

import main
import mas_arena.agents.bench_agent as bench_agent_module
from mas_arena.agents.base import create_agent_system
from mas_arena.utils.openai_compat import normalize_openai_api_base


def test_parse_tool_list_arg():
    assert main.parse_tool_list_arg(None) is None
    assert main.parse_tool_list_arg("") == []
    assert main.parse_tool_list_arg("none") == []
    assert main.parse_tool_list_arg("python_interpreter, ALL") == [
        "python_interpreter",
        "ALL",
    ]


def test_normalize_openai_api_base():
    assert normalize_openai_api_base("https://example.com", "https://api.openai.com/v1") == "https://example.com/v1"
    assert normalize_openai_api_base("https://example.com/custom/", "https://api.openai.com/v1") == "https://example.com/custom"


def test_bench_agent_reads_tool_config_from_registry(monkeypatch):
    class DummyCodeAgent:
        pass

    class DummyToolCallingAgent:
        pass

    class DummyLLM:
        async def aclose(self):
            return None

    def fake_initialize_model(self):
        self.llm = DummyLLM()

    def fake_initialize_tools(self):
        self.manager_tools = []
        self.search_tools = []

    def fake_create_agents(self):
        return {
            "workers": [
                SimpleNamespace(name="manager", agent=DummyCodeAgent(), llm=self.llm),
                SimpleNamespace(name="search", agent=DummyToolCallingAgent(), llm=self.llm),
            ]
        }

    monkeypatch.setattr(bench_agent_module, "CodeAgent", DummyCodeAgent)
    monkeypatch.setattr(bench_agent_module, "ToolCallingAgent", DummyToolCallingAgent)
    monkeypatch.setattr(bench_agent_module.BenchAgent, "_initialize_model", fake_initialize_model)
    monkeypatch.setattr(bench_agent_module.BenchAgent, "_initialize_tools", fake_initialize_tools)
    monkeypatch.setattr(bench_agent_module.BenchAgent, "_create_agents", fake_create_agents)

    agent = create_agent_system(
        "bench_agent",
        config={
            "model_name": "qwen3-32b",
            "manager_tools": ["python_interpreter", "final_answer"],
            "search_tools": ["ALL"],
            "evaluator": "gaia",
        },
    )

    assert agent.config["model_name"] == "qwen3-32b"
    assert agent.manager_tools_config == ["python_interpreter", "final_answer"]
    assert agent.search_tools_config == ["ALL"]
    assert agent.benchmark_name == "gaia"
