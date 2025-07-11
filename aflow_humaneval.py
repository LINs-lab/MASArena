import os
from dotenv import load_dotenv

from mas_arena.core.openai_llm import OpenAILLM
from mas_arena.core.model_configs import OpenAILLMConfig
from mas_arena.evaluators.humaneval_evaluator import HumanEvalEvaluator
from mas_arena.optimizers.aflow_optimizer import AFlowOptimizer
from mas_arena.configs.aflow.aflow_config import EXPERIMENTAL_CONFIG

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_BASE=os.getenv("ANTHROPIC_API_BASE")

def main():
    optimizer_config = OpenAILLMConfig(model="claude-3-7-sonnet-latest", openai_key=ANTHROPIC_API_KEY,base_url = ANTHROPIC_API_BASE)
    optimizer_llm = OpenAILLM(config=optimizer_config)
    executor_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY,base_url = OPENAI_API_BASE)
    executor_llm = OpenAILLM(config=executor_config)

    # load evaluator
    humaneval = HumanEvalEvaluator("humaneval",{})

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path = "mas_arena/configs/aflow",
        optimized_path = "example/aflow/humaneval/optimization4",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=1,
        eval_rounds=1,
        max_rounds=3,
        **EXPERIMENTAL_CONFIG["humaneval"]
    )

    optimizer.setup()
    # run optimization
    optimizer.optimize(humaneval)

    # run test
    optimizer.test(humaneval)


if __name__ == "__main__":
    main()