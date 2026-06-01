import uuid
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from typing import Any
import json

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from mas_arena.train.verl.smolagent import SmolagentsAgent
from mas_arena.memory.llm import GPTChat
from mas_arena.memory.melo_memory import MELOMemory
from mas_arena.memory.common import MASMessage
from mas_arena.memory.utils import EmbeddingFunc
from mas_arena.train.verl.common import to_agent_loop_output

logger = logging.getLogger(__name__)
import asyncio


def load_problems_from_jsonl(jsonl_path):
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    problems.append(json.loads(line))
                except Exception as e:
                    print(f"Error parsing line: {e}")
    return problems


class MemoryAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent: SmolagentsAgent = None
        self.memory_manager: MELOMemory = None
        print("MemoryAgentLoop initialized, agent will be created on first run.")

    async def get_llm_server_address(self, server_name: str = None) -> str:
        """Dynamically retrieves the LLM server address from VERL's server manager."""
        server = self.server_manager._choose_server(server_name or uuid.uuid4().hex)
        base_url = await server.get_server_address.remote()
        base_url = f"http://{base_url}/v1"
        logger.info(f"get_server_address#base_url: {base_url}")
        return base_url

    async def get_llm_server_model_name(self):
        """Dynamically retrieves the model name from VERL's config."""
        model_name = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
        logger.info(f"get_server_model_name#model_name: {model_name}")
        return model_name

    async def _initialize_agent_and_memory(self):
        """Initializes the agent and memory manager using config from VERL on the first run."""
        if self.agent is None:
            print("First run: Initializing SmolAgent and MELOMemory for RL training...")

            model_name = await self.get_llm_server_model_name()
            api_base = await self.get_llm_server_address()

            print(f"Connecting to model: {model_name} at {api_base}")

            # Pass the model config to SmolAgent.
            # The API key is a dummy key for local vLLM servers, as they don't require authentication.

            embed_func = EmbeddingFunc(
                os.getenv(
                    "MAS_RL_EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            )
            persist_dir = os.getenv("RL_MELO_PERSIST_DIR", "mas_arena/train/persis/melo")

            agent_config = {"model_name": model_name, "api_base": api_base}
            model = GPTChat(
                model_name=model_name,
            )

            self.agent = SmolagentsAgent(config=agent_config)
            self.memory_manager = MELOMemory(
                namespace="melo_memory",
                llm_model=model,
                embedding_func=embed_func,
                persist_dir=persist_dir,
            )
            self.agent.meta_memory = self.memory_manager
            print("Initialization complete.")

    async def run(self, messages: list, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Rollout loop.
        """
        # Lazily initialize agent and memory on the first run to use async methods for setup
        await self._initialize_agent_and_memory()

        ground_truth = kwargs.get("ground_truth", "")  # Passed from the data loader

        # Build augmented prompt from task_prompt (similar to normalized_problem in benchmark_runner.py)

        # 3. Agent Execution
        # The agent's `evaluate` method should return results dict (awaited like in benchmark_runner.py)
        results = await self.agent.run_agent(problem)
        final_answer = results.get("final_answer", "")
        trajectory = results.get("intermediate_steps", [])
        search_keywords = results.get("search_keywords", [])

        # 4. Determine Outcome for Memory Storage
        is_correct = ground_truth in final_answer

        # 5. Memory Writing
        # After the episode, the experience is summarized and stored.
        mas_message = MASMessage(
            task_search_keywords=augumented_prompt,
            task_question=augumented_prompt,
            task_trajectory=str(trajectory),
            label=is_correct,
            extra_fields={},
        )
        self.memory_manager.add_memory(mas_message)

        # 6. Prepare output for VERL
        # The 'trajectory' from your agent should be a list of OpenAI-formatted messages
        # We also pass necessary info to the reward function via `extra_info`
        # NOTE: This assumes VERL's trainer passes `extra_info` to the reward function.
        # This may require a minor patch to VERL or a different design.
        # A simpler way is to re-calculate reward inside the loop, but using the
        # external reward function is more standard.
        output = await to_agent_loop_output(
            tokenizer=self.tokenizer,
            messages=trajectory,
            response_length=self.config.actor_rollout_ref.rollout.response_length,
        )

        # Pass crucial data to the reward function via the metrics dict
        output.metrics["extra_info"] = {
            "trajectory": trajectory,
            "memory_manager": self.memory_manager,
            "task_prompt": task_prompt,
        }

        return output


async def main():
    # 初始化 agent 和 memory
    loop = MemoryAgentLoop()
    await loop._initialize_agent_and_memory()

    # 加载所有 problem
    problems = load_problems_from_jsonl("../../data/gaia_test.jsonl")
    for idx, problem in enumerate(problems):
        print(f"\n===== Running problem {idx + 1}/{len(problems)}: {problem.get('task_id', '')} =====")
        # 兼容字段
        prompt_problem = {
            "problem": problem.get("Question", ""),
            "files": [problem["file_name"]] if problem.get("file_name") else [],
            "solution": problem.get("Final answer", ""),
        }
        result = await loop.agent.run_agent(prompt_problem)
        print(f"Final answer: {result.get('final_answer')}")
        print(f"Intermediate steps: {result.get('intermediate_steps')}")


if __name__ == "__main__":
    asyncio.run(main())
