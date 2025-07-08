import time
import json
import os
import asyncio
import re
import tempfile
import subprocess
import sys
from collections import defaultdict, deque
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, TypedDict, Any, List

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


# Prompt Templates for EvoMAC
INITIAL_CODING_ROLE = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'."""

INITIAL_CODING = """Here is a function completion task:

Task: "{task}".

Please think step by step and complete the function.

Your answer must include the complete function implementation in a Python code block. Use the following format:

```python
def function_name(parameters):
    \"\"\"
    Function description
    \"\"\"
    # Your implementation here
    return result
```

Make sure to:
1. Include the complete function definition
2. Implement all required logic
3. Return the correct result
4. Do not include test cases or main() function"""

ORGANIZER = """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'.

You are Chief Technology Officer. we are both working at EvoMAC. We share a common interest in collaborating to successfully complete a task assigned by a new customer.

You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology ("IT") staff members to perform everyday operations.

Here is a new customer's task: {task}.

To complete the task, You must write a response that appropriately solves the requested instruction based on your expertise and customer's needs."""

ORGANIZING = """Here is a function completion task:

Task: "{task}".

Programming Language: "{language}"

The implemention of the task(source codes) are: "{codes}"

Your goal is to organize a coding team to complete the function completion task.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks. 

### COMPOSITION

```
Task 1: Task 1 description
Task 2: Task 2 description
```

### WORKFLOW

```
Task 1: []
Task 2: [Task 1]
```

Please note that the decomposition should be both effective and efficient.

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) The WORKFLOW should not contain circles!
3) The programmer number and the task number should be as small as possible.
4) Your task should not include anything related to testing, writing document or computation cost optimizing."""

SUBCODECOMPLETE = """Here is a function completion task:
Task: "{task}".
Programming Language: "{language}"
The implemention of the task(source codes) are: "{codes}"
I will give you a subtask below, you should carefully read the subtask and do the following things: 
1) If the subtask is a specific task related to the function completion, please think step by step and reason yourself to finish the task.
2) If the subtask is a test report of the code, please check the source code and the test report, and then think step by step and reason yourself to fix the bug. 
Subtask description: "{subtask}"
3) You should output the COMPLETE code content. Use the following format:

```python
def function_name(parameters):
    \"\"\"
    Function description
    \"\"\"
    # Your implementation here
    return result
```

Make sure to:
1. Include the complete function definition
2. Implement all required logic
3. Return the correct result
4. Do not include test cases or main() function
5. No placeholder code (such as 'pass' in Python)"""

TESTORGANIZING="""According to the function completion requirements listed below: 

Task: "{task}".

Programming Language: "{language}"

Your goal is to organize a testing team to complete the function completion task.

There are one default tasks: 

1) use some simplest case to test the logic. The case must be as simple as possible, and you should ensure every 'assert' you write is 100% correct

Follow the format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. 

### COMPOSITION

```
Task 1: Task 1 description
Task 2: Task 2 description
```

### WORKFLOW

```
Task 1: []
Task 2: [Task 1]
```

Note that:

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) DO NOT include things like implement the code in your task description.
3) The task number should be as small as possible. Only one task is also acceptable."""

TESTCODECOMPLETE = """According to the function completion requirements listed below: 
Task: "{task}".
Please locate the example test case given in the function definition, these test case will be used latter.
The implemention of the function is:
"{codes}"
Testing Task description: "{subtask}"
According to example test case in the Task description, please write these test cases to locate the bugs. You should not add any other testcases except for the example test case given in the Task description
The output must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is "{test_file_name}", "LANGUAGE" in the programming language,"REQUIREMENTS" is the targeted requirement of the test case, and "CODE" is the test code that is used to test the specific requirement of the file. Format:

FILENAME
```LANGUAGE
'''
REQUIREMENTS
'''
CODE
```
You will start with the "{test_file_name}" and finish the code follows in the strictly defined format.
Please note that:
1) The code should be fully functional. No placeholders (such as 'pass' in Python).
2) You should write the test file with 'unittest' python library. Import the functions you need to test if necessary.
3) The test case should be as simple as possible, and the test case number should be less than 5.
4) According to example test case in the Task description, please only write these test cases to locate the bugs. You should not add any other testcases by yourself except for the example test case given in the Task description"""

UPDATING = """Here is a function completion task:

Task:

{task}.

Source Codes:

{codes}

Current issues: 

{issues}.

According to the task, source codes and current issues given above, design a programmmer team to solve current issues.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "Workflow" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks.

### COMPOSITION

```
Programmer 1: Task 1 description
Programmer 2: Task 2 description
```

### WORKFLOW

```
Programmer 1: []
Programmer 2: [Programmer 1]
```

Please note that:

1) You should repeat exactly the current issues in the task description of module COMPOSITION in a line. For example: Programmer 1: AssertionError: function_name(input) != expected_output. The actual output is: actual_output.

2) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].

3) The WORKFLOW should not contain circles!

4) The programmer number and the task number should be as small as possible. One programmer is also acceptable.

5) DO NOT include things like implement the code in your task description."""


class Codes:
    """Manages code content and formatting"""
    
    def __init__(self):
        self.codes = {}
        
    def _update_codes(self, response: str):
        """Update codes from LLM response using robust extraction methods"""
        code = self._extract_code_robust(response)
        if code:
            # Use a default filename for the extracted code
            self.codes['solution.py'] = code
    
    def _extract_code_robust(self, text: str) -> str:
        """
        Extract Python code from text using multiple fallback methods,
        similar to HumanEval evaluator's approach.
        """
        # Method 1: Look for "## Validated Code" section (from other agents)
        qa_match = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if qa_match:
            return qa_match.group(1).strip()
        
        # Method 2: Look for any ```python``` fenced block
        block_match = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if block_match:
            return block_match.group(1).strip()
        
        # Method 3: Look for function definition patterns
        fn_match = re.search(r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\n{2,}|\Z)", text)
        if fn_match:
            return fn_match.group(1).strip()
        
        # Method 4: Try original filename + code pattern as fallback
        pattern = r'([a-z_]+\.py)\s*\n\s*```python\s*(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0][1].strip()
        
        # Method 5: Last resort - try to find any code-like content
        # Look for lines that start with 'def ' or contain python-like syntax
        lines = text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if 'def ' in line and '(' in line and ')' in line and ':' in line:
                in_code_block = True
                code_lines.append(line)
            elif in_code_block:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t') and 'def ' not in line:
                    # Likely end of function
                    break
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # If all else fails, return empty string
        return ""
    
    def _get_codes(self) -> str:
        """Get formatted codes for display"""
        if not self.codes:
            return "No codes available."
        
        formatted = []
        for filename, content in self.codes.items():
            formatted.append(f"{filename}:\n```python\n{content}\n```")
        return "\n\n".join(formatted)
    
    def _get_raw_codes(self) -> str:
        """Get raw code content"""
        if not self.codes:
            return ""
        
        # Return the main implementation
        if len(self.codes) == 1:
            return list(self.codes.values())[0]
        
        # If multiple files, concatenate them
        return "\n\n".join(self.codes.values())
    
    def _format_code(self, code: str) -> str:
        """Format code content"""
        return code.strip()


class Organization:
    """Manages workflow organization and task decomposition"""
    
    def __init__(self):
        self.composition = {}
        self.workflow = {}
    
    def _update_orgs(self, response: str):
        """Update organization from LLM response"""
        try:
            # Extract COMPOSITION
            comp_match = re.search(r'COMPOSITION:\s*(.*?)(?=WORKFLOW:|$)', response, re.DOTALL)
            if comp_match:
                comp_text = comp_match.group(1).strip()
                self.composition = {}
                for line in comp_text.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        task_match = re.match(r'- (Task_?\d+): (.+)', line)
                        if task_match:
                            task_name = task_match.group(1)
                            task_desc = task_match.group(2)
                            self.composition[task_name] = task_desc
            
            # Extract WORKFLOW
            workflow_match = re.search(r'WORKFLOW:\s*(.*)', response, re.DOTALL)
            if workflow_match:
                workflow_text = workflow_match.group(1).strip()
                self.workflow = {}
                for line in workflow_text.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        task, deps = line.split(':', 1)
                        task = task.strip()
                        deps_str = deps.strip()
                        if deps_str.startswith('[') and deps_str.endswith(']'):
                            deps_content = deps_str[1:-1].strip()
                            if deps_content:
                                dependencies = [d.strip() for d in deps_content.split(',')]
                            else:
                                dependencies = []
                        else:
                            dependencies = []
                        self.workflow[task] = dependencies
        except Exception as e:
            print(f"Error parsing organization: {e}")
            # Fallback to simple task structure
            self.composition = {"Task_1": "Complete the implementation"}
            self.workflow = {"Task_1": []}
    
    def _format_orgs(self) -> str:
        """Format organization structure"""
        result = "COMPOSITION:\n"
        for task, desc in self.composition.items():
            result += f"- {task}: {desc}\n"
        
        result += "\nWORKFLOW:\n"
        for task, deps in self.workflow.items():
            result += f"{task}: {deps}\n"
        
        return result
    
    def _format_composition(self) -> Dict[str, str]:
        """Get composition dictionary"""
        return self.composition.copy()
    
    def _format_workflow(self) -> Dict[str, List[str]]:
        """Get workflow dictionary"""
        return self.workflow.copy()


class EvoMAC(AgentSystem):
    """
    EvoMAC (Evolutionary Multi-Agent Coding) Agent System
    
    A multi-agent coding framework that simulates a software company organization
    with role-based task decomposition, workflow execution, testing, and iterative optimization.
    """
    
    def __init__(self, name: str = "evomac", config: Dict[str, Any] = None):
        """Initialize EvoMAC system"""
        super().__init__(name, config)
        
        self.config = config or {}
        self.iteration = self.config.get("iteration", 5)
        self.language = self.config.get("language", "python")
        
        # Initialize code and organization managers
        self.codes = Codes()
        self.test_codes = Codes()
        self.organization = Organization()
        self.test_organization = Organization()
        
        # Initialize LLM client
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Validate API configuration
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        
        # ChatOpenAI automatically reads from environment variables
        self.client = ChatOpenAI(
            model=self.model_name,
            temperature=0.7
        )
    
    def format_messages(self, role: str, content: str) -> List[Dict[str, str]]:
        """Format messages for LLM"""
        return [
            {"role": "system", "content": role},
            {"role": "user", "content": content}
        ]
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM and return response"""
        try:
            # Convert dict messages to LangChain message objects
            from langchain_core.messages import SystemMessage, HumanMessage
            
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = self.client.invoke(
                langchain_messages,
                config={"temperature": 0.7}
            )
            return response.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
    
    def topological_sort(self, workflow: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on workflow dependencies"""
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)
        
        # Build adjacency list and in-degree count
        for node, dependencies in workflow.items():
            for dep in dependencies:
                adj_list[dep].append(node)
                in_degree[node] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([node for node in workflow if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            current_node = queue.popleft()
            topo_order.append(current_node)
            
            # Process neighbors
            for neighbor in adj_list[current_node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(topo_order) != len(workflow):
            raise ValueError("The workflow contains a cycle and cannot be topologically sorted.")
        
        return topo_order
    
    def execute_workflow(self, query: str):
        """Execute the main coding workflow"""
        try:
            composition = self.organization._format_composition()
            workflow = self.organization._format_workflow()
            
            if not composition or not workflow:
                return
            
            # Get execution order via topological sort
            execution_order = self.topological_sort(workflow)
            
            # Execute each task in order
            for task_name in execution_order:
                if task_name in composition:
                    subtask_desc = composition[task_name]
                    
                    # Generate subtask completion prompt
                    prompt = SUBCODECOMPLETE.format(
                        task=query,
                        language=self.language,
                        codes=self.codes._get_codes(),
                        subtask=subtask_desc
                    )
                    
                    messages = self.format_messages("Programmer", prompt)
                    response = self.call_llm(messages)
                    
                    # Update codes with new implementation
                    self.codes._update_codes(response)
        
        except Exception as e:
            print(f"Workflow execution failed: {e}")
    
    def test_bugs(self, test_code: str) -> tuple[bool, str]:
        """Test code for bugs and return results"""
        try:
            # Create temporary file for testing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                test_file = f.name
            
            try:
                # Run tests using subprocess
                if sys.platform.startswith('win'):
                    result = subprocess.run([
                        sys.executable, '-m', 'unittest', test_file.replace('.py', '')
                    ], capture_output=True, text=True, timeout=30, cwd=os.path.dirname(test_file))
                else:
                    result = subprocess.run([
                        'python3', '-m', 'unittest', test_file
                    ], capture_output=True, text=True, timeout=30)
                
                # Check if tests passed
                has_bugs = result.returncode != 0
                test_report = result.stdout + result.stderr
                
                return has_bugs, test_report
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(test_file)
                except:
                    pass
                    
        except Exception as e:
            return True, f"Test execution failed: {str(e)}"
    
    def execute_test_workflow(self, query: str) -> tuple[bool, str]:
        """Execute test workflow and return bug status and reports"""
        try:
            # Generate test organization
            test_organizing_prompt = TESTORGANIZING.format(
                task=query,
                codes=self.codes._get_codes()
            )
            
            messages = self.format_messages(
                ORGANIZER.format(task=query),
                test_organizing_prompt
            )
            test_organization = self.call_llm(messages)
            self.test_organization._update_orgs(test_organization)
            
            # Execute test tasks
            test_composition = self.test_organization._format_composition()
            test_workflow = self.test_organization._format_workflow()
            
            if not test_composition:
                return False, "No tests to execute"
            
            # Generate test code for each test task
            all_test_reports = []
            has_any_bugs = False
            
            for task_name in test_composition:
                test_task_desc = test_composition[task_name]
                
                # Generate test code
                test_prompt = TESTCODECOMPLETE.format(
                    task=query,
                    language=self.language,
                    codes=self.codes._get_codes(),
                    subtask=test_task_desc,
                    test_file_name="test_solution.py"
                )
                
                messages = self.format_messages("Test Engineer", test_prompt)
                test_response = self.call_llm(messages)
                
                # Extract and run test code
                test_code_match = re.search(r'```python\s*(.*?)```', test_response, re.DOTALL)
                if test_code_match:
                    test_code = test_code_match.group(1).strip()
                    
                    # Combine source code and test code
                    combined_code = self.codes._get_raw_codes() + "\n\n" + test_code
                    
                    # Run tests
                    has_bugs, test_report = self.test_bugs(combined_code)
                    if has_bugs:
                        has_any_bugs = True
                    
                    all_test_reports.append(f"Test {task_name}: {test_report}")
            
            return has_any_bugs, "\n\n".join(all_test_reports)
            
        except Exception as e:
            return True, f"Test workflow execution failed: {str(e)}"
    
    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the EvoMAC agent system on a given problem.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including messages and final answer
        """
        query = problem["problem"]
        all_messages = []
        
        try:
            # 1. Initial coding
            initial_coding_messages = self.format_messages(
                INITIAL_CODING_ROLE,
                INITIAL_CODING.format(task=query)
            )
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in initial_coding_messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = self.client.invoke(
                langchain_messages,
                config={"temperature": 0.7}
            )
            
            initial_coding_codes = response.content
            self.codes._update_codes(initial_coding_codes)
            
            # Record message with usage metadata
            ai_message = {
                'content': initial_coding_codes,
                'name': 'initial_coder',
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': getattr(response, 'usage_metadata', None)
            }
            all_messages.append(ai_message)
            
            # 2. Generate workflow organization
            organizing_messages = self.format_messages(
                ORGANIZER.format(task=query),
                ORGANIZING.format(
                    task=query,
                    language=self.language,
                    codes=self.codes._get_codes()
                )
            )
            
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in organizing_messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = self.client.invoke(
                langchain_messages,
                config={"temperature": 0.7}
            )
            
            organization = response.content
            self.organization._update_orgs(organization)
            
            # Record message
            ai_message = {
                'content': organization,
                'name': 'organizer',
                'role': 'assistant',
                'message_type': 'ai_response',
                'usage_metadata': getattr(response, 'usage_metadata', None)
            }
            all_messages.append(ai_message)
            
            # 3. Execute workflow and test
            self.execute_workflow(query)
            has_bug_in_tests, test_reports = self.execute_test_workflow(query)
            
            # 4. Iterative optimization
            for i in range(self.iteration - 1):
                if not has_bug_in_tests:
                    break
                
                # Update workflow to solve problems
                updating_messages = self.format_messages(
                    ORGANIZER.format(task=query),
                    UPDATING.format(
                        task=query,
                        codes=self.codes._get_codes(),
                        issues=test_reports
                    )
                )
                
                # Convert messages to LangChain format
                langchain_messages = []
                for msg in updating_messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                
                response = self.client.invoke(
                    langchain_messages,
                    config={"temperature": 0.7}
                )
                
                organization = response.content
                self.organization._update_orgs(organization)
                
                # Record message
                ai_message = {
                    'content': organization,
                    'name': f'updater_iteration_{i+1}',
                    'role': 'assistant',
                    'message_type': 'ai_response',
                    'usage_metadata': getattr(response, 'usage_metadata', None)
                }
                all_messages.append(ai_message)
                
                # Re-execute workflow and tests
                self.execute_workflow(query)
                has_bug_in_tests, test_reports = self.execute_test_workflow(query)
            
            final_answer = self.codes._get_raw_codes()
            
            return {
                "messages": all_messages,
                "final_answer": final_answer
            }
            
        except Exception as e:
            error_message = {
                'content': f"EvoMAC execution failed: {str(e)}",
                'name': 'error_handler',
                'role': 'assistant',
                'message_type': 'error',
                'usage_metadata': None
            }
            all_messages.append(error_message)
            
            return {
                "messages": all_messages,
                "final_answer": f"Error: {str(e)}"
            }


# Register the EvoMAC agent system
AgentSystemRegistry.register(
    "evomac",
    EvoMAC,
    iteration=5,
    language="python"
)

