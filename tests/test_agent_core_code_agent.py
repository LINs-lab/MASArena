from types import SimpleNamespace

from smolagents import Tool

from mas_arena.agents.agent_core.agents import CodeAgent


class DummyModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.monitor = SimpleNamespace(total_input_token_count=0, total_output_token_count=0)

    def __call__(self, messages):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "<answer>fallback</answer>"


class AsyncEchoTool(Tool):
    name = "async_echo"
    description = "Echo text asynchronously."
    inputs = {
        "text": {
            "type": "string",
            "description": "Text to echo.",
        }
    }
    output_type = "string"

    async def forward(self, text: str) -> str:
        return f"async:{text}"


def test_code_agent_sync_wrapper_resolves_async_tools():
    model = DummyModel(
        [
            '```python\nprint(async_echo("hello"))\n```',
            "<answer>done</answer>",
        ]
    )
    agent = CodeAgent(tools=[AsyncEchoTool()], model=model, max_steps=1, verbosity_level=0)

    assert agent.run("test async tool") == "done"
    observations = [
        obs
        for step in agent.memory.get_steps()
        for obs in getattr(step, "observations", [])
    ]
    assert "async:hello" in observations
    assert not any("coroutine object" in str(obs) for obs in observations)


def test_code_agent_forces_final_answer_after_max_steps_and_flags_repeated_errors():
    repeated_bad_code = "```python\nprint(missing_name)\n```"
    model = DummyModel(
        [
            repeated_bad_code,
            repeated_bad_code,
            "<answer>best guess</answer>",
        ]
    )
    agent = CodeAgent(tools=[], model=model, max_steps=2, verbosity_level=0)

    assert agent.run("test repeated failure") == "best guess"
    force_call_messages = model.calls[-1]
    assert "same error has already occurred" in force_call_messages[-2]["content"]
    assert "maximum step budget" in force_call_messages[-1]["content"]
