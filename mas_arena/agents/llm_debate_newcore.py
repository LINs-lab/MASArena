"""Two homogeneous BenchAgent debaters alternate for three rounds, then aggregate."""
from typing import Any, Dict

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.workflow_protocol import bench_settings, model_name, task_text


class LLMDebate(AgentSystem):
    def __init__(self, name: str = "llm_debate", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.agents_num = self.config.get("agents_num", 2)
        self.rounds_num = self.config.get("rounds_num", 3)
        self.model_name = model_name(self.config)
        self.base_system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
        self.bench_agent_params = bench_settings(self.config)
        self.bench_agent_params["additional_instructions"] = "\n\n".join(filter(None, [
            self.base_system_prompt, self.config.get("additional_instructions"), self.format_prompt,
        ]))
        self.debate_agents = [BenchAgent(name=f"debate_agent_{index + 1}", **self.bench_agent_params)
                              for index in range(self.agents_num)]
        self.aggregator_agent = BenchAgent(name="aggregator_bench_agent", **self.bench_agent_params)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        query = task_text(problem)
        history, messages = [], []
        answers = [""] * self.agents_num
        for round_index in range(self.rounds_num):
            for index, agent in enumerate(self.debate_agents):
                prompt = (
                    f"Can you solve the following problem? {query}\n"
                    "Please explain your reasoning step by step and state your final answer clearly.\n\n"
                    "Recent opinions from the debate:\n" + ("\n\n".join(history) or "No previous answers.") +
                    "\n\nUse these opinions carefully as additional advice and provide an updated answer."
                )
                try:
                    result = await agent.run_agent_step(prompt, {"id": problem.get("id", ""), "round": round_index + 1})
                    answer = result.get("final_answer", "No answer generated.")
                    for message in result.get("messages", []):
                        if isinstance(message, dict):
                            message = {**message, "name": f"debate_agent_{index + 1}", "round": round_index + 1}
                        messages.append(message)
                except Exception as exc:
                    answer = f"Debate agent {index + 1} failed: {exc}"
                    messages.append({"role": "assistant", "content": answer, "message_type": "error"})
                answers[index] = answer
                history.append(f"Agent {index + 1} (Round {round_index + 1}): {answer}")

        result = await self.aggregator_agent.run_agent_step(
            f"Task:\n{query}\n\nDebate history:\n" + "\n\n".join(history) +
            f"\n\nSynthesize the debate and provide a final answer.\n{self.format_prompt}",
            {"id": problem.get("id", ""), "is_aggregation": True},
        )
        messages.extend(result.get("messages", []))
        answer = result.get("final_answer", "No answer generated.")
        return {"messages": messages, "final_answer": answer, "extracted_answer": answer,
                "agent_responses": answers, "debate_history": history,
                "rounds_completed": self.rounds_num, "agents_participated": self.agents_num}

    async def aclose(self):
        for agent in [*self.debate_agents, self.aggregator_agent]:
            await agent.aclose()


AgentSystemRegistry.register("llm_debate", LLMDebate, agents_num=2, rounds_num=3)
