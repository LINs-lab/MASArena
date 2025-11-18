# Acknowledgement: Modified from AFlow (https://github.com/geekan/MetaGPT/blob/main/metagpt/ext/aflow/scripts/optimizer.py) under MIT License

import asyncio
import os
import re
import shutil
from typing import Any, List, Union, Optional

import numpy as np
from pydantic import Field
from tqdm import tqdm

from mas_arena.agent_flow.workflow_evaluator import EvaluationUtils
from mas_arena.core_serializer.operators import LLmOptimizeOutput
from mas_arena.optimizers.optimizer import Optimizer
from mas_arena.agents import AgentSystem
from mas_arena.utils.llm_parser import LLMOutputParser
from mas_arena.utils.convergence_utils import ConvergenceUtils
from mas_arena.utils.data_utils import DataUtils
from mas_arena.utils.experience_utils import ExperienceUtils
from mas_arena.utils.graph_utils import GraphUtils, OPERATOR_MAP
from mas_arena.evaluators.base_evaluator import BaseEvaluator

class AFlowOptimizer(Optimizer):
    """
    Implements an iterative, evolutionary optimization process for workflows
    based on performance feedback, driven by an optimizer LLM.
    """
    question_type: str = Field(
        description="The type of question to optimize the workflow for, e.g., qa, match, code, etc.")
    graph_path: str = Field(
        description="The folder of the initial workflow graph. This folder must contain `graph.py` and `prompt.py` files.")
    optimized_path: str | None = Field(
        default=None,
        description="The root directory to save all optimization rounds and results. Defaults to `graph_path` if not provided.")
    initial_round: int = Field(
        default=0,
        description="The round number to start or continue optimization from.")
    operators: List[str] = Field(
        default_factory=lambda: list(OPERATOR_MAP.keys()),
        description="The operators available for the optimizer LLM to use.")
    sample: int = Field(default=4, description="The number of rounds to sample from the top scores.")
    max_rounds: int = Field(
        default=20, description="The maximum number of optimization rounds.")
    validation_rounds: int = Field(
        default=5,
        description="The number of times to run a workflow on the validation set to get a stable performance score.")
    eval_rounds: int = Field(
        default=3,
        description="The number of times to run the final best workflow on the test set for final evaluation.")
    check_convergence: bool = Field(
        default=True, description="Whether to stop early if performance plateaus.")
    optimizer_agent: Union[AgentSystem, None] = Field(default=None, description="The agent system for the optimize")

    executor_agent: Union[AgentSystem, None] = Field(default=None, description="The agent system for the execute")
    train_size: int = Field(
        default=20,
        description="The size of the training set for evaluation.")
    test_size: int = Field(
        default=40,
        description="The size of the test set for evaluation.")
    def setup(self, **kwargs):
        """Initializes the optimizer, sets up paths, and prepares utilities."""
        self.root_path = self.optimized_path or self.graph_path
        os.makedirs(self.root_path, exist_ok=True)

        # Initialize utility handlers
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

        self.graph = None
        self.round = self.initial_round

        if self.optimizer_agent is None:
            raise ValueError("optimizer_agent must be provided.")
        if self.executor_agent is None:
            self.executor_agent = self.optimizer_agent

        self._prepare_initial_round_files()

    def optimize(self, evaluator: BaseEvaluator):
        """Runs the main optimization loop for a given benchmark."""
        self.evaluator = evaluator
        for _ in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(self._run_with_retries(self._perform_one_optimization_round))

            self.round += 1
            if self._has_converged() or self.round >= self.max_rounds:
                break

    def test(self, evaluator: BaseEvaluator, test_rounds: List[int] | None = None):
        """Runs a final test evaluation on the best or specified workflow rounds."""
        self.evaluator = evaluator
        if test_rounds is None:
            best_round = self.find_best_performing_round()
            print(f"No test rounds provided, using best round: {best_round}")
            test_rounds = [best_round]

        for _ in tqdm(range(self.eval_rounds), desc="Final Evaluation"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._execute_final_evaluation(test_rounds))

    async def _perform_one_optimization_round(self) -> float:
        num_validation_runs = self.validation_rounds
        benchmark_data = self.data_utils.load_results(self.root_path)

        if self.round == 0:
            return await self._run_baseline_evaluation(num_validation_runs, benchmark_data)
        return await self._generate_and_evaluate_new_workflow(num_validation_runs, benchmark_data)

    async def _run_baseline_evaluation(self, num_validation_runs: int, benchmark_data: list) -> float:
        self.graph_utils.create_round_directory(self.root_path, self.round)
        self.graph = self.graph_utils.load_graph(self.round, self.root_path)
        print(f"Running baseline evaluation for round {self.round}...")
        avg_score = await self.evaluation_utils.evaluate_graph_async(self, num_validation_runs, benchmark_data, initial=True)
        print(f"Baseline score for round {self.round}: {avg_score}")
        return avg_score

    async def _generate_and_evaluate_new_workflow(self, num_validation_runs: int, benchmark_data: list) -> float:
        next_round_num = self.round + 1
        new_workflow_dir = self.graph_utils.create_round_directory(self.root_path, next_round_num)

        while True:
            parent_workflow_sample = self._select_parent_workflow()
            prompt_template, graph_code = self.graph_utils.read_graph_files(parent_workflow_sample["round"],
                                                                            self.root_path)
            solve_graph_code = self.graph_utils.extract_solve_graph(graph_code)

            processed_experience = self.experience_utils.load_experience()
            experience_prompt = self.experience_utils.format_experience(processed_experience,
                                                                        parent_workflow_sample["round"])

            if self.optimizer_agent is None:
                raise ValueError("Optimizer agent is not initialized.")
            operator_description = self.graph_utils.load_operators_description(self.operators, self.optimizer_agent)
            log_data = self.data_utils.load_log(parent_workflow_sample["round"])

            # Generate the prompt to ask the LLM for a modification
            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience_prompt, parent_workflow_sample["score"], solve_graph_code[0], prompt_template,
                operator_description, self.question_type, log_data
            )

            # Get and parse the LLM's response
            response = await self.optimizer_agent.run_agent(problem={"problem": graph_optimize_prompt},
                                                            parse_mode="str")
            output: Optional[LLMOutputParser] = response.get("final_answer")
            if isinstance(output, list):
                raise TypeError(f"Expected a single LLMOutputParser, but got a list.")

            print(f"-round:{self.round}-- Optimizer LLM Response ---")
            print(output.content)
            try:
                parsed_response = LLmOptimizeOutput.parse(output.content, parse_mode="xml")
                response = parsed_response.get_structured_data()
                print("Parsed LLM Optimization response successfully.")
            except Exception:
                response = self._parse_llm_optimization_output(output.content,
                                                               orig_graph=solve_graph_code[0],
                                                               orig_prompt=prompt_template)
                print("Failed to parse LLM Optimization response, using regex fallback.")
            if self.experience_utils.check_modification(processed_experience, response['modification'],parent_workflow_sample["round"]):
                break

        avg_score = await self._evaluate_and_record_new_workflow(
            new_workflow_dir, response, parent_workflow_sample, benchmark_data, num_validation_runs
        )
        print(f"Score for new workflow in round {self.round}: {avg_score}")
        return avg_score

    async def _evaluate_and_record_new_workflow(self, directory: str, llm_response: dict, parent_sample: dict,
                                                benchmark_data: list, num_validation_runs: int):
        """Saves the newly generated workflow files, evaluates it, and records the experience."""
        # Write the new graph.py and prompt.py to the next round's directory
        self.graph_utils.write_graph_files(directory, llm_response)

        experience_entry = self.experience_utils.create_experience_data(parent_sample, llm_response['modification'])

        # Load the newly created graph to run it
        self.graph = self.graph_utils.load_graph(self.round + 1, self.root_path)

        # Evaluate the new graph's performance on the validation set
        avg_score = await self.evaluation_utils.evaluate_graph_async(self, num_validation_runs, benchmark_data,
                                                                     initial=False, train_size=self.train_size,
                                                                     test_size=self.test_size)

        # Save the results of this experiment
        self.experience_utils.update_experience(directory, experience_entry, avg_score)

        return avg_score

    async def _execute_final_evaluation(self, test_rounds: List[int]):
        """Runs the final evaluation on the test set for the specified rounds."""
        print("Running final test evaluation...")
        benchmark_data = self.data_utils.load_results(self.root_path)
        json_file_path = self.data_utils.get_results_file_path(self.root_path)
        scores = []

        for round_num in test_rounds:
            print(f"Testing workflow from round {round_num}...")
            self.graph = self.graph_utils.load_graph(round_num, self.root_path)
            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test_async(self)
            scores.append(score)

            new_result_entry = self.data_utils.create_result_data(round_num, score, avg_cost, total_cost)
            benchmark_data.append(new_result_entry)

            print(f"  - Score: {score}, Avg Cost: {avg_cost}, Total Cost: {total_cost}")
            self.data_utils.save_results(json_file_path, benchmark_data)

        avg_final_score = np.mean(scores)
        print(f"\nAverage score across all test runs: {avg_final_score}")
        return avg_final_score


    def _prepare_initial_round_files(self):
        """Ensures the files for the starting round (usually round 0) are in place."""
        if self.round == 0:
            round_zero_path = os.path.join(self.root_path, f"round_{self.round}")
            os.makedirs(round_zero_path, exist_ok=True)
            # Copy the initial graph and prompt files to the round_0 directory
            if not os.path.exists(os.path.join(round_zero_path, "graph.py")):
                shutil.copy2(os.path.join(self.graph_path, "graph.py"), os.path.join(round_zero_path, "graph.py"))
            if not os.path.exists(os.path.join(round_zero_path, "prompt.py")):
                shutil.copy2(os.path.join(self.graph_path, "prompt.py"), os.path.join(round_zero_path, "prompt.py"))
            # Update imports in the copied graph.py
            self.graph_utils.update_prompt_import(os.path.join(round_zero_path, "graph.py"), round_zero_path)

        if not os.path.exists(os.path.join(self.root_path, f"round_{self.round}")):
            raise ValueError(f"Starting round {self.round} does not exist in {self.root_path}")

    async def _run_with_retries(self, func: callable, max_retries: int = 3) -> Any:
        """A wrapper to execute a function with a retry mechanism on failure."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                print(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
                if attempt + 1 == max_retries:
                    print("Max retries reached. Stopping this round.")
                    return None
                await asyncio.sleep(5 * (attempt + 1))
        return None

    def _select_parent_workflow(self) -> dict:
        """Selects a high-performing workflow from previous rounds to be the basis for a new modification."""
        top_rounds = self.data_utils.get_top_rounds(self.sample)
        return self.data_utils.select_round(top_rounds)

    def _parse_llm_optimization_output(self, content: str, orig_graph: str, orig_prompt: str) -> dict:
        """Parses the LLM's output, trying structured parsing first and falling back to regex."""
        try:
            # First, try the Pydantic-based XML parser
            parsed_data = LLmOptimizeOutput.parse(content, parse_mode="xml")
            return parsed_data.get_structured_data()
        except Exception:
            # If Pydantic parsing fails, fall back to regex-based extraction
            response = {"modification": "", "graph": "", "prompt": ""}

            modification_match = re.search(r'<modification>(.*?)</modification>', content, re.DOTALL)
            if modification_match:
                response["modification"] = modification_match.group(1).strip()

            # Find all python code blocks
            code_blocks = re.finditer(r'```(?:python)?(.*?)```', content, re.DOTALL)
            for block in code_blocks:
                code = block.group(1).strip()
                if 'class' in code or 'workflow' in code.lower():
                    response["graph"] = code  # type: ignore
                else:
                    response["prompt"] = code # Assume other code blocks are prompts

            # If parsing fails catastrophically, return original to avoid crash
            if not response["graph"] and not response["prompt"]:
                response["modification"] = "No modification due to error in LLM output"
                response["graph"] = orig_graph
                response["prompt"] = orig_prompt

            return response

    def _has_converged(self) -> bool:
        """Checks if the optimization process has converged and prints results if it has."""
        if not self.check_convergence:
            return False

        converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)
        if converged:
            print(f"Convergence detected at round {convergence_round}, stopping at round {final_round}.")
            self.convergence_utils.print_results()
            return True
        return False

    def find_best_performing_round(self) -> int:
        """Loads all scores and returns the round number with the highest score."""
        ranked_scores = self.data_utils._load_scores()
        if not ranked_scores:
            raise RuntimeError("No scores found. Cannot determine the best round.")
        return ranked_scores[0]["round"]