import mas_arena.core.operators as operator
from mas_arena.core.llm_utils import create_llm_instance
from mas_arena.core.model_configs import LLMConfig
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from . import prompt as prompt_custom


class Workflow:

    def __init__(
            self,
            name: str,
            llm_config: LLMConfig,
            evaluator: BaseEvaluator
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.evaluator = evaluator
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # await self.custom(input=, instruction="")
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point,
                                                   instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        return solution['response']

