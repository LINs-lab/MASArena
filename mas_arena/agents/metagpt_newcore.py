import time
import os
import re
import json
from typing import Dict, List, Any
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

@dataclass
class Agent:
    name: str
    description: str
    goals: List[str]
    constraints: List[str]
    role: str
    system_prompt: str

    bench_agent: BenchAgent = field(init=False, repr=False)
    memory: Dict[str, Any] = None 
    

    def __init__(
        self,
        name: str,
        description: str,
        goals: List[str],
        constraints: List[str],
        role: str,
        system_prompt: str,
        bench_agent: BenchAgent, 
        memory: Dict[str, Any] = None,
    ):
        self.name = name
        self.description = description
        self.goals = goals
        self.constraints = constraints
        self.role = role
        self.system_prompt = system_prompt
        self.bench_agent = bench_agent  
        if memory is None:
            self.memory = {"messages": [], "knowledge": {}, "tasks": [], "completed_tasks": []}
        else:
            self.memory = memory


    def add_to_memory(self, key: str, value: Any):
        if key not in self.memory:
            self.memory[key] = []
        self.memory[key].append(value)

    def get_from_memory(self, key: str) -> Any:
        return self.memory.get(key, [])

    def clear_memory(self, key: str = None):
        if key:
            self.memory[key] = []
        else:
            self.memory = {"messages": [], "knowledge": {}, "tasks": [], "completed_tasks": []}

class MetaGPT(AgentSystem):
    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        super().__init__(name, config)

        self.bench_agent_executor = BenchAgent(
            model=self.config.get("model_name", "gpt-4o-mini"),
            api_key=self.config.get("api_key"),
            api_base=self.config.get("api_base"),
            search_max_steps=self.config.get("search_max_steps", 10),
            verbosity_level=self.config.get("verbosity_level", 2),
            manager_tools= self.config.get("manager_tools"),
            search_tools= self.config.get("search_tools"),
            memory= self.config.get("memory"),
            additional_instructions=self.config.get("additional_instructions")
        )
        
        self._create_agents()
        self.message_queue = []
        self.task_status = {
            "current_task": None,
            "task_history": [],
            "iteration_count": 0,
            "max_iterations": self.config.get("max_iterations", 3),
        }
        self.message_history = []
        
    def _create_agents(self) -> Dict[str, List[Agent]]:
        agents_dict = {}


        bench_agent = self.bench_agent_executor 

        # Product Manager
        agents_dict["product_manager"] = Agent(
            name="Product Manager",
            description="Analyzes requirements for code generation tasks.",
            goals=["Extract functional requirements", "Define task scope", "Parse constraints and examples"],
            constraints=["Must align with problem prompt", "Ensure clarity"],
            role="product_manager",
            system_prompt="""You are a Product Manager for code generation tasks. Your task is to analyze the problem prompt and extract functional requirements, constraints, and examples from the docstring.

<plan>
1. Read the problem prompt, including the function signature and docstring.
2. Extract the function signature (name, parameters, return type).
3. Parse the docstring to identify input/output requirements, constraints, and examples.
4. List functional requirements (e.g., input types, output format, edge cases).
5. Define the task scope clearly, including any constraints or specific behaviors.
</plan>

For HumanEval problems, the prompt includes a function signature and docstring with examples and constraints. Extract:
- Function name and parameters
- Input constraints (e.g., value ranges, types)
- Output format (e.g., string, integer, float)
- Example inputs and outputs

Output in plain text with markdown formatting, wrapped in <answer> tags:

<answer>
## Task Title
{task_id}

## Function Signature
{Function name, parameters, and return type}

## Description
{Problem description from docstring}

## Requirements
- {Requirement 1}
- {Requirement 2}

## Constraints
- {Constraint 1}
- {Constraint 2}

## Examples
- Input: {Example input 1}, Output: {Example output 1}
- Input: {Example input 2}, Output: {Example output 2}

## Scope
- {Scope description}
</answer>
""", 
            bench_agent=bench_agent,
        )

        # Architect
        agents_dict["architect"] = Agent(
            name="Architect",
            description="Designs implementation approach for code generation.",
            goals=["Define implementation strategy", "Outline solution structure"],
            constraints=["Must use Python", "Keep solution simple and maintainable"],
            role="architect",
            system_prompt="""You are an Architect for code generation tasks. Your task is to design the implementation approach based on the Product Manager's requirements.

<plan>
1. Review the Product Manager's requirements, constraints, examples, and scope.
2. Select Python as the technology stack.
3. Outline the function's logic in pseudocode, addressing all requirements and examples.
4. Define the solution structure (e.g., functions, data structures) to handle inputs and produce the expected output.
</plan>

Output in plain text with markdown formatting, wrapped in <answer> tags:

<answer>
## Implementation Strategy
{Pseudocode or logic description addressing requirements and examples}

## Solution Structure
- {Structure point 1}
- {Structure point 2}

## Technology Stack
- Python
</answer>
""", 
            bench_agent=bench_agent,
        )

        # Project Manager
        agents_dict["project_manager"] = Agent(
            name="Project Manager",
            description="Breaks down code generation tasks and assigns them.",
            goals=["Assign coding task", "Ensure task clarity"],
            constraints=["Single task for Engineer", "Align with architecture"],
            role="project_manager",
            system_prompt="""You are a Project Manager for code generation tasks. Your task is to assign the coding task to the Engineer based on requirements and architecture.

<plan>
1. Review Product Manager's requirements, constraints, examples, and Architect's design.
2. Define a single, clear task for the Engineer to implement the function, including the function signature and expected behavior.
3. Ensure the task aligns with the architecture, requirements, and examples.
</plan>

Output in plain text with markdown formatting, wrapped in <answer> tags:

<answer>
## Task Assignment
- Task ID: code_function
- Description: {Implement the function description}
- Assigned to: Engineer
- Function Signature: {Function signature}
- Requirements: {List key requirements}
- Constraints: {List key constraints}
- Examples: {List key examples}
- Architecture Notes: {Key architecture points}
</answer>
""", 
            bench_agent=bench_agent,
        )

        # Engineer
        agents_dict["engineer"] = Agent(
            name="Engineer",
            description="Responsible for code writing and implementation, developing according to architect's design and project manager's task assignments.",
            goals=["Write correct Python code", "Implement features", "Fix bugs", "Optimize performance"],
            constraints=["Must follow architecture design", "Must match function signature", "Use markdown code block"],
            role="engineer",
            system_prompt="""You are a professional Engineer responsible for code writing and implementation, developing according to the Architect's design and Project Manager's task assignments.

<plan>
1. Review Product Manager's requirements, Architect's design, and Project Manager's task assignment, including function signature, constraints, and examples.
2. Write Python code matching the function signature in the prompt.
3. Implement all required features, handle edge cases, and ensure the code produces outputs matching the examples.
4. Optimize code for performance and readability.
5. Provide a clear explanation of the code logic.
</plan>

Output in plain text with markdown formatting, wrapped in <answer> tags:

<answer>
## Code
```python
{Python code here}
```

## Implementation Details
- {Explanation point 1}
- {Explanation point 2}

## Features Implemented
- {Feature 1}
- {Feature 2}

## Optimizations
- {Optimization 1 or "None"}
</answer>
""", 
            bench_agent=bench_agent,
        )

        # QA Engineer
        agents_dict["qa_engineer"] = Agent(
            name="QA Engineer",
            description="Tests and validates Python code, ensuring correctness and compliance with requirements.",
            goals=["Validate code", "Output final code", "Identify and fix bugs"],
            constraints=["Ensure code correctness", "Match requirements", "Provide clear feedback"],
            role="qa_engineer",
            system_prompt="""You are a QA Engineer for code generation tasks. Your task is to test the Engineer's code against requirements, constraints, examples, and provided test cases, and provide the final validated Python code.

<plan>
1. Review Product Manager's requirements, constraints, examples, Architect's design, and Engineer's code.
2. Validate the code against the function signature, requirements, and examples.
3. Execute the provided test cases to check for correctness, including edge cases.
4. Identify bugs or issues (e.g., incorrect logic, missing edge cases).
5. Provide fixes if bugs are found, or confirm code correctness.
6. Output the final validated code in a markdown block.
</plan>

Output in plain text with markdown formatting, wrapped in <answer> tags:

<answer>
## Test Results
- {Test case 1 description}: {Pass/Fail}
- {Test case 2 description}: {Pass/Fail}

## Bugs Found
- {Bug 1 or "None"}

## Fixes Applied
- {Fix 1 or "None"}

## Validated Code
```python
{Final validated Python code}
```
</answer>
""", 
            bench_agent=bench_agent,
        )

        self.agents = agents_dict
        return {"workers": list(agents_dict.values())}
        

    def _publish_message(self, from_agent: str, to_agent: str, message_type: str, content: Any):
        message = {
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "content": content,
            "timestamp": time.time(),
        }
        self.message_queue.append(message)
        if to_agent in self.agents:
            self.agents[to_agent].add_to_memory("messages", message)
            
    def _subscribe_messages(self, agent_name: str, message_type: str = None) -> List[Dict[str, Any]]:
        messages = []
        for message in self.message_queue:
            if message["to"] == agent_name and (message_type is None or message["type"] == message_type):
                messages.append(message)
        return messages
    
    async def _run_agent_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.agents[agent_name]
        

        augmented_question = f"{agent.system_prompt}\n\nTask:\n{str(task)}"
        

        additional_args = {"agent_name": agent_name}
        

        result = await agent.bench_agent.run_agent_step(
            augmented_question=augmented_question,
            additional_args=additional_args
        )
        
        content = result.get("content", result.get("final_answer", ""))
        usage_metadata = result.get("usage_metadata")


        task_result = {"content": content, "agent": agent_name}


        ai_message = type('AIMessageMock', (object,), {
            'content': content, 
            'name': agent_name, 
            'usage_metadata': usage_metadata,
        })()
        self.message_history.append(ai_message)

        self._publish_message(
            agent_name, "system", "task_result", {"content": task_result, "usage_metadata": usage_metadata}
        )

        return task_result
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        max_iterations = self.task_status["max_iterations"]
        iteration = 0
        result = None

        while iteration < max_iterations:
            self.task_status["iteration_count"] = iteration
            self.task_status["current_task"] = task

            product_manager_result = await self._run_agent_task("product_manager", task)
            architect_result = await self._run_agent_task("architect", {"product_manager_result": product_manager_result})
            project_manager_result = await self._run_agent_task(
                "project_manager",
                {"product_manager_result": product_manager_result, "architect_result": architect_result},
            )
            developer_result = await self._run_agent_task(
                "engineer",
                {
                    "product_manager_result": product_manager_result,
                    "architect_result": architect_result,
                    "project_manager_result": project_manager_result,
                },
            )
            tester_result = await self._run_agent_task(
                "qa_engineer",
                {
                    "product_manager_result": product_manager_result,
                    "architect_result": architect_result,
                    "project_manager_result": project_manager_result,
                    "developer_result": developer_result,
                    "test_cases": task.get("test", ""),
                },
            )

            result = {
                "product_manager_result": product_manager_result,
                "architect_result": architect_result,
                "project_manager_result": project_manager_result,
                "developer_result": developer_result,
                "tester_result": tester_result,
            }

            qa_content = tester_result.get("content", "")
            if not self._need_iteration(qa_content):
                break

            task["qa_feedback"] = {
                "bugs": self._extract_bugs(qa_content),
                "suggestions": self._extract_suggestions(qa_content),
            }
            iteration += 1

        return result

    def _extract_bugs(self, qa_content: str) -> List[str]:
        bugs_match = re.search(r"## Bugs Found\n- ([^\n]+)", qa_content)
        if bugs_match and bugs_match.group(1) != "None":
            return [bugs_match.group(1)]
        return []

    def _extract_suggestions(self, qa_content: str) -> List[str]:
        suggestions_match = re.search(r"## Improvement Suggestions\n- ([^\n]+)", qa_content)
        if suggestions_match and suggestions_match.group(1) != "None":
            return [suggestions_match.group(1)]
        return []

    def _need_iteration(self, qa_content: str) -> bool:
        return bool(self._extract_bugs(qa_content) or self._extract_suggestions(qa_content))

    async def run_agent(self, task_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a task.
        
        Args:
            task_data: The task data in a standardized format (prepared by evaluator)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing execution results and messages
        """
        start = time.time()
        self.message_history = []

        try:
            result_chain = await self._process_task(task_data)

            # Return the raw QA output without specific extraction
            tester_out = result_chain.get("tester_result", {}).get("content", "")

            messages = [
                m for m in self.message_history 
                if hasattr(m, "usage_metadata") and m.usage_metadata is not None
            ]

            return {
                "result": result_chain,
                "execution_time": time.time() - start,
                "messages": messages,
                "final_answer": tester_out,
            }

        except Exception as exc:
            return {
                "result": {"error": str(exc)},
                "execution_time": 0,
                "messages": [],
                "final_answer": f"Error: {exc}",
            }


AgentSystemRegistry.register("metagpt", MetaGPT, max_iterations=3)