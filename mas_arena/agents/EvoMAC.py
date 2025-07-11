"""
EvoMAC (Evolutionary Multi-Agent Coding) Agent System

A sophisticated multi-agent coding framework that simulates a software company organization
with role-based task decomposition, workflow execution, testing, and iterative optimization.

This system includes:
- Chief Technology Officer (CTO) for task decomposition
- Programmers for code implementation  
- Test managers and engineers for quality assurance
- Updaters for iterative improvement

Authors: EvoMAC Development Team
"""


import os
import asyncio
import re
import tempfile
import sys
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables for configuration
load_dotenv()

# SYSTEM PROMPTS - Define roles and behaviors for each agent type
AGENT_SYSTEM_PROMPTS = {
    "cto": """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'.

You are Chief Technology Officer. we are both working at EvoMAC. We share a common interest in collaborating to successfully complete a task assigned by a new customer.

You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology ("IT") staff members to perform everyday operations.

Your goal is to organize a coding team to complete the function completion task.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "WORKFLOW" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks. 

COMPOSITION:
Task 1: Task 1 description
Task 2: Task 2 description

WORKFLOW:
Task 1: []
Task 2: [Task 1]

Please note that the decomposition should be both effective and efficient.

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) The WORKFLOW should not contain circles!
3) The programmer number and the task number should be as small as possible.
4) Your task should not include anything related to testing, writing document or computation cost optimizing.""",

    "initial_coder": """EvoMAC is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'.""",

    "programmer": """You are a professional programmer at EvoMAC. Your task is to implement code according to the given specifications.

<plan>
1. Carefully read the task requirements and any provided code context
2. Think step by step about the implementation approach
3. Write clean, efficient Python code that solves the problem
4. Ensure your code handles all edge cases and requirements
5. Return the complete function implementation
</plan>

Make sure to:
1. Include the complete function definition
2. Implement all required logic
3. Return the correct result
4. Do not include test cases or main() function
5. No placeholder code (such as 'pass' in Python)""",

    "test_organizer": """You are a Test Manager at EvoMAC. Your goal is to organize a testing team to complete the function testing task.

There are one default tasks: 

1) use some simplest case to test the logic. The case must be as simple as possible, and you should ensure every 'assert' you write is 100% correct

Follow the format: "COMPOSITION" is the composition of tasks, and "WORKFLOW" is the workflow of the programmers. 

COMPOSITION:
Task 1: Task 1 description
Task 2: Task 2 description

WORKFLOW:
Task 1: []
Task 2: [Task 1]

Note that:

1) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].
2) DO NOT include things like implement the code in your task description.
3) The task number should be as small as possible. Only one task is also acceptable.""",

    "test_engineer": """You are a Test Engineer at EvoMAC. Your task is to write test cases according to the function specifications.

The output must strictly follow a markdown code block format, where the following tokens must be replaced such that "FILENAME" is "{test_file_name}", "LANGUAGE" in the programming language,"REQUIREMENTS" is the targeted requirement of the test case, and "CODE" is the test code that is used to test the specific requirement of the file. Format:

FILENAME
```LANGUAGE
'''
REQUIREMENTS
'''
CODE
```

Please note that:
1) The code should be fully functional. No placeholders (such as 'pass' in Python).
2) You should write the test file with 'unittest' python library. Import the functions you need to test if necessary.
3) The test case should be as simple as possible, and the test case number should be less than 5.
4) According to example test case in the Task description, please only write these test cases to locate the bugs. You should not add any other testcases by yourself except for the example test case given in the Task description""",

    "updater": """You are a Senior Developer at EvoMAC. Your task is to organize a programmer team to solve current issues in the code.

You should follow the following format: "COMPOSITION" is the composition of tasks, and "WORKFLOW" is the workflow of the programmers. Each task is assigned to a programmer, and the workflow shows the dependencies between tasks.

COMPOSITION:
Programmer 1: Task 1 description
Programmer 2: Task 2 description

WORKFLOW:
Programmer 1: []
Programmer 2: [Programmer 1]

Please note that:

1) You should repeat exactly the current issues in the task description of module COMPOSITION in a line. For example: Programmer 1: AssertionError: function_name(input) != expected_output. The actual output is: actual_output.

2) The WORKFLOW is to show the relationship between each task. You should not answer any specific task in [].

3) The WORKFLOW should not contain circles!

4) The programmer number and the task number should be as small as possible. One programmer is also acceptable.

5) DO NOT include things like implement the code in your task description."""
}


# TASK PROMPT TEMPLATES - Define specific task instructions
TASK_PROMPT_TEMPLATES = {
    "initial_coding": """Here is a function completion task:

Task: "{task}".

Please think step by step and complete the function.

Make sure to:
1. Include the complete function definition
2. Implement all required logic
3. Return the correct result
4. Do not include test cases or main() function""",

    "task_organization": """Here is a function completion task:

Task: "{task}".

Programming Language: "{language}"

The implementation of the task(source codes) are: "{codes}"

{format_prompt}""",

    "subtask_completion": """Here is a function completion task:
Task: "{task}".
Programming Language: "{language}"
The implementation of the task(source codes) are: "{codes}"
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

{format_prompt}""",

    "test_organization": """According to the function completion requirements listed below: 

Task: "{task}".

Programming Language: "{language}"

{format_prompt}""",

    "test_code_completion": """According to the function completion requirements listed below: 
Task: "{task}".
Please locate the example test case given in the function definition, these test case will be used latter.
The implementation of the function is:
"{codes}"
Testing Task description: "{subtask}"
According to example test case in the Task description, please write these test cases to locate the bugs. You should not add any other testcases except for the example test case given in the Task description

You will start with the "{test_file_name}" and finish the code follows in the strictly defined format.

{format_prompt}""",

    "issue_resolution": """Here is a function completion task:

Task: {task}.

Source Codes: {codes}

Current issues: {issues}.

According to the task, source codes and current issues given above, design a programmer team to solve current issues.

{format_prompt}"""
}


# CODE MANAGEMENT CLASS - Handles code extraction, storage, and formatting
class CodeManager:
    """
    Manages code content extraction, storage, and formatting.
    
    This class provides robust methods for extracting Python code from LLM responses
    using multiple fallback strategies, similar to HumanEval evaluator approaches.
    """
    
    def __init__(self):
        """Initialize code storage."""
        self.codes: Dict[str, str] = {}
        self.default_filename = "solution.py"
    
    def update_from_response(self, response: str) -> None:
        """
        Update codes from LLM response using robust extraction methods.
        
        Args:
            response: Raw LLM response text containing code
        """
        extracted_code = self._extract_code_with_fallbacks(response)
        if extracted_code:
            self.codes[self.default_filename] = extracted_code
    
    def _extract_code_with_fallbacks(self, text: str) -> str:
        """
        Extract Python code from text using multiple fallback methods.
        
        This method tries several extraction strategies in order of preference:
        1. Look for "## Validated Code" section (from other agents)
        2. Extract from ```python``` fenced blocks
        3. Find function definition patterns
        4. Parse filename + code patterns  
        5. Last resort: find any Python-like syntax
        
        Args:
            text: Input text containing code
            
        Returns:
            Extracted code string, or empty string if no code found
        """
        # Strategy 1: Look for validated code sections
        validated_match = re.search(
            r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", 
            text, 
            re.IGNORECASE
        )
        if validated_match:
            code = validated_match.group(1).strip()
            return self._clean_extracted_code(code)
        
        # Strategy 2: Extract from standard Python code blocks
        block_match = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if block_match:
            code = block_match.group(1).strip()
            return self._clean_extracted_code(code)
        
        # Strategy 3: Find function definition patterns
        function_match = re.search(
            r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\n{2,}|\Z)", 
            text
        )
        if function_match:
            code = function_match.group(1).strip()
            return self._clean_extracted_code(code)
        
        # Strategy 4: Parse filename + code patterns (legacy support)
        filename_pattern = r'([a-z_]+\.py)\s*\n\s*```python\s*(.*?)```'
        filename_matches = re.findall(filename_pattern, text, re.DOTALL)
        if filename_matches:
            code = filename_matches[0][1].strip()
            return self._clean_extracted_code(code)
        
        # Strategy 5: Last resort - find Python-like content
        return self._extract_python_like_content(text)
    
    def _clean_extracted_code(self, code: str) -> str:
        """
        Clean extracted code to fix common syntax issues.
        
        Args:
            code: Raw extracted code
            
        Returns:
            Cleaned code string
        """
        if not code:
            return ""
        
        # 确保代码以函数定义开始，如果不是则尝试找到函数定义
        if not code.strip().startswith('def '):
            # 尝试找到第一个函数定义
            lines = code.split('\n')
            start_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    start_idx = i
                    break
            
            if start_idx >= 0:
                code = '\n'.join(lines[start_idx:])
            else:
                # 如果没有找到函数定义，返回原代码
                return code.strip()
        
        # 修复三引号docstring问题
        lines = code.split('\n')
        cleaned_lines = []
        in_triple_quote = False
        quote_type = None
        quote_start_line = -1
        
        for i, line in enumerate(lines):
            # 检测三引号的开始和结束
            if '"""' in line:
                if not in_triple_quote:
                    # 开始三引号字符串
                    in_triple_quote = True
                    quote_type = '"""'
                    quote_start_line = i
                    cleaned_lines.append(line)
                else:
                    # 结束三引号字符串
                    if quote_type == '"""':
                        in_triple_quote = False
                        quote_type = None
                        cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
            elif "'''" in line:
                if not in_triple_quote:
                    # 开始三引号字符串
                    in_triple_quote = True
                    quote_type = "'''"
                    quote_start_line = i
                    cleaned_lines.append(line)
                else:
                    # 结束三引号字符串
                    if quote_type == "'''":
                        in_triple_quote = False
                        quote_type = None
                        cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        # 如果docstring未正确关闭，添加关闭标记
        if in_triple_quote and quote_type:
            # 找到正确的缩进级别
            indent = ""
            if quote_start_line >= 0 and quote_start_line < len(cleaned_lines):
                # 获取docstring开始行的缩进
                start_line = cleaned_lines[quote_start_line]
                if '"""' in start_line or "'''" in start_line:
                    # 如果docstring在同一行开始，使用相同缩进
                    indent = re.match(r'^(\s*)', start_line).group(1)
                else:
                    # 否则找到函数体的缩进
                    for line in cleaned_lines:
                        if line.strip().startswith('def '):
                            # 找到下一个非空行的缩进
                            func_idx = cleaned_lines.index(line)
                            for j in range(func_idx + 1, len(cleaned_lines)):
                                next_line = cleaned_lines[j]
                                if next_line.strip():
                                    indent = re.match(r'^(\s*)', next_line).group(1)
                                    break
                            break
            
            # 添加关闭的三引号
            cleaned_lines.append(f'{indent}{quote_type}')
            print(f"[DEBUG] Added missing closing quote: {quote_type}")
        
        cleaned_code = '\n'.join(cleaned_lines)
        
        # 检查是否需要添加必要的导入语句
        cleaned_code = self._add_missing_imports(cleaned_code)
        
        # 移除多余的空行
        cleaned_code = re.sub(r'\n{3,}', '\n\n', cleaned_code)
        
        return cleaned_code.strip()
    
    def _add_missing_imports(self, code: str) -> str:
        """
        Add missing import statements that are commonly needed.
        
        Args:
            code: Code to check for missing imports
            
        Returns:
            Code with necessary imports added
        """
        if not code:
            return code
        
        lines = code.split('\n')
        imports_to_add = []
        
        # 检查是否使用了typing相关的类型提示
        if any(re.search(r'\b(List|Dict|Tuple|Optional|Union|Any)\[', line) or re.search(r'\bAny\b', line) for line in lines):
            if not any('from typing import' in line or 'import typing' in line for line in lines):
                # 确定需要导入的类型
                needed_types = set()
                for line in lines:
                    for type_hint in ['List', 'Dict', 'Tuple', 'Optional', 'Union']:
                        if re.search(rf'\b{type_hint}\[', line):
                            needed_types.add(type_hint)
                    # 单独检查Any，因为它可能不带方括号
                    if re.search(r'\bAny\b', line):
                        needed_types.add('Any')
                
                if needed_types:
                    imports_to_add.append(f"from typing import {', '.join(sorted(needed_types))}")
        
        # 如果需要添加导入，将它们插入到函数定义之前
        if imports_to_add:
            # 找到第一个函数定义的位置
            func_start_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    func_start_idx = i
                    break
            
            if func_start_idx >= 0:
                # 在函数定义前插入导入语句
                new_lines = (lines[:func_start_idx] + 
                           imports_to_add + 
                           [''] +  # 空行分隔
                           lines[func_start_idx:])
                return '\n'.join(new_lines)
            else:
                # 如果没有找到函数定义，在开头添加导入
                return '\n'.join(imports_to_add + [''] + lines)
        
        return code
    
    def _extract_python_like_content(self, text: str) -> str:
        """
        Last resort extraction method for Python-like content.
        
        Args:
            text: Input text
            
        Returns:
            Extracted Python-like code or empty string
        """
        lines = text.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            # Start of function definition
            if 'def ' in line and '(' in line and ')' in line and ':' in line:
                in_function = True
                code_lines.append(line)
            elif in_function:
                # Continue collecting function lines
                if line.strip() and not line.startswith((' ', '\t')) and 'def ' not in line:
                    # Likely end of function (non-indented line)
                    break
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip() if code_lines else ""
    
    def get_formatted_codes(self) -> str:
        """
        Get formatted codes for display purposes.
        
        Returns:
            Formatted string with filename and code blocks
        """
        if not self.codes:
            return "No codes available."
        
        formatted_sections = []
        for filename, content in self.codes.items():
            formatted_sections.append(f"{filename}:\n```python\n{content}\n```")
        
        return "\n\n".join(formatted_sections)
    
    def get_raw_code(self) -> str:
        """
        Get raw code content without formatting.
        
        Returns:
            Raw code string, concatenated if multiple files exist
        """
        if not self.codes:
            return ""
        
        if len(self.codes) == 1:
            return list(self.codes.values())[0]
        
        # If multiple files exist, concatenate them
        return "\n\n".join(self.codes.values())
    
    def has_code(self) -> bool:
        """Check if any code has been stored."""
        return bool(self.codes and any(code.strip() for code in self.codes.values()))


# WORKFLOW ORGANIZATION CLASS - Manages task decomposition and dependencies
class WorkflowOrganizer:
    """
    Manages workflow organization and task decomposition.
    
    This class handles parsing of task compositions and workflow dependencies
    from LLM responses, ensuring proper task organization and dependency management.
    """
    
    def __init__(self):
        """Initialize workflow organization structures."""
        self.composition: Dict[str, str] = {}
        self.workflow: Dict[str, List[str]] = {}
    
    def update_from_response(self, response: str) -> None:
        """
        Update organization from LLM response.
        
        Parses COMPOSITION and WORKFLOW sections from the response text.
        
        Args:
            response: LLM response containing organization structure
        """
        try:
            self._parse_composition(response)
            self._parse_workflow(response)
            
            if not self.composition or not self.workflow:
                print("Warning: Empty composition or workflow, using fallback")
                self._set_fallback_organization()
                
        except Exception as e:
            print(f"Warning: Failed to parse organization structure: {e}")
            self._set_fallback_organization()
    
    def _parse_composition(self, response: str) -> None:
        """
        Parse COMPOSITION section from response.
        
        Args:
            response: LLM response text
        """
        # 首先尝试标准的COMPOSITION格式
        composition_match = re.search(
            r'COMPOSITION[:\s]*\n?\s*(.*?)(?=WORKFLOW:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        
        # 如果标准格式失败，尝试其他格式
        if not composition_match:
            # 尝试查找<answer>标签内的COMPOSITION
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_content = answer_match.group(1)
                composition_match = re.search(
                    r'COMPOSITION[:\s]*\n?\s*(.*?)(?=WORKFLOW:|$)', 
                    answer_content, 
                    re.DOTALL | re.IGNORECASE
                )
        
        # 如果还是没有找到，检查是否是纯文本响应需要创建单个任务
        if not composition_match:
            # 检查响应是否看起来像一个实现或解释
            if ('implement' in response.lower() or 'function' in response.lower() or 
                'test' in response.lower() or 'case' in response.lower()):
                # 创建单个任务
                self.composition = {"Task_1": "Complete the implementation based on requirements"}
                return
        
        if composition_match:
            comp_text = composition_match.group(1).strip()
            self.composition = {}
            
            # 移除可能的markdown代码块标记
            comp_text = re.sub(r'^```.*?\n', '', comp_text, flags=re.MULTILINE)
            comp_text = re.sub(r'\n```$', '', comp_text)
            
            for line in comp_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # 支持多种格式：
                # 1. "- Task_1: description" (原格式)
                # 2. "Task 1: description" (CTO prompt格式)
                # 3. "Programmer 1: description" (updater格式)
                task_match = None
                
                if line.startswith('- '):
                    # 原格式: "- Task_1: description"
                    task_match = re.match(r'- (Task_?\d+): (.+)', line)
                elif ':' in line and not line.endswith(':'):
                    # 新格式: "Task 1: description" 或 "Programmer 1: description"
                    # 但确保不是单独的"COMPOSITION:"这样的标题行
                    task_match = re.match(r'((?:Task|Programmer)\s*\d+):\s*(.+)', line)
                    
                if task_match:
                    task_name = task_match.group(1).replace(' ', '_')  # 统一格式：Task_1
                    task_desc = task_match.group(2).strip()
                    # 过滤掉无效的任务描述
                    if task_desc and task_desc != '[]' and len(task_desc) > 2:
                        self.composition[task_name] = task_desc
        
        # 如果没有解析到有效的composition，输出调试信息
        if not self.composition:
            print(f"[DEBUG] Failed to parse composition. Response preview: {response[:300]}...")
    
    def _parse_workflow(self, response: str) -> None:
        """
        Parse WORKFLOW section from response.
        
        Args:
            response: LLM response text
        """
        # 首先尝试标准的WORKFLOW格式
        workflow_match = re.search(r'WORKFLOW[:\s]*\n?\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        
        # 如果标准格式失败，尝试其他格式
        if not workflow_match:
            # 尝试查找<answer>标签内的WORKFLOW
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_content = answer_match.group(1)
                workflow_match = re.search(r'WORKFLOW[:\s]*\n?\s*(.*)', answer_content, re.DOTALL | re.IGNORECASE)
        
        # 如果还是没有找到，为composition中的任务创建简单的workflow
        if not workflow_match and self.composition:
            # 为现有的composition创建简单的线性workflow
            self.workflow = {}
            task_names = list(self.composition.keys())
            for i, task_name in enumerate(task_names):
                if i == 0:
                    self.workflow[task_name] = []
                else:
                    self.workflow[task_name] = [task_names[i-1]]
            return
        
        if workflow_match:
            workflow_text = workflow_match.group(1).strip()
            self.workflow = {}
            
            # 移除可能的markdown代码块标记
            workflow_text = re.sub(r'^```.*?\n', '', workflow_text, flags=re.MULTILINE)
            workflow_text = re.sub(r'\n```$', '', workflow_text)
            
            for line in workflow_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    task, deps_str = line.split(':', 1)
                    task = task.strip().replace(' ', '_')  # 统一格式：Task_1
                    deps_str = deps_str.strip()
                    
                    # Parse dependencies from [dep1, dep2] format
                    dependencies = self._parse_dependencies(deps_str)
                    # 同样统一依赖项的格式
                    dependencies = [dep.replace(' ', '_') for dep in dependencies]
                    self.workflow[task] = dependencies
    
    def _parse_dependencies(self, deps_str: str) -> List[str]:
        """
        Parse dependency string into list of dependencies.
        
        Args:
            deps_str: Dependency string like "[Task1, Task2]" or "[]"
            
        Returns:
            List of dependency task names
        """
        if deps_str.startswith('[') and deps_str.endswith(']'):
            deps_content = deps_str[1:-1].strip()
            if deps_content:
                deps = [dep.strip() for dep in deps_content.split(',')]
                # 统一格式化依赖项名称 (Task 1 -> Task_1)
                normalized_deps = []
                for dep in deps:
                    if re.match(r'Task\s+\d+', dep):
                        dep = dep.replace(' ', '_')
                    elif re.match(r'Programmer\s+\d+', dep):
                        dep = dep.replace(' ', '_')
                    normalized_deps.append(dep)
                return normalized_deps
        return []
    
    def _set_fallback_organization(self) -> None:
        """Set fallback organization structure when parsing fails."""
        self.composition = {"Task_1": "Complete the implementation"}
        self.workflow = {"Task_1": []}
    
    def get_formatted_structure(self) -> str:
        """
        Get formatted organization structure for display.
        
        Returns:
            Formatted string with composition and workflow
        """
        result = "COMPOSITION:\n"
        for task, desc in self.composition.items():
            result += f"- {task}: {desc}\n"
        
        result += "\nWORKFLOW:\n"
        for task, deps in self.workflow.items():
            result += f"{task}: {deps}\n"
        
        return result
    
    def get_composition(self) -> Dict[str, str]:
        """Get composition dictionary copy."""
        return self.composition.copy()
    
    def get_workflow(self) -> Dict[str, List[str]]:
        """Get workflow dictionary copy."""
        return self.workflow.copy()
    
    def has_valid_structure(self) -> bool:
        """Check if organization has valid structure."""
        return bool(self.composition and self.workflow)


# DEPENDENCY RESOLVER - Handles topological sorting of workflow tasks
class DependencyResolver:
    """
    Utility class for resolving task dependencies using topological sorting.
    
    Ensures tasks are executed in the correct order based on their dependencies,
    while detecting and preventing circular dependencies.
    """
    
    @staticmethod
    def get_execution_order(workflow: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort on workflow dependencies.
        
        Args:
            workflow: Dictionary mapping task names to their dependencies
            
        Returns:
            List of task names in execution order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        if not workflow:
            return []
        
        # Build adjacency list and calculate in-degrees
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)
        
        # Initialize in-degree for all nodes
        for node in workflow:
            in_degree[node] = 0
        
        # Build graph and count in-degrees
        for node, dependencies in workflow.items():
            for dep in dependencies:
                adj_list[dep].append(node)
                in_degree[node] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([node for node in workflow if in_degree[node] == 0])
        execution_order = []
        
        # Process nodes in topological order
        while queue:
            current_node = queue.popleft()
            execution_order.append(current_node)
            
            # Update neighbors
            for neighbor in adj_list[current_node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Detect cycles
        if len(execution_order) != len(workflow):
            raise ValueError("Circular dependency detected in workflow")
        
        return execution_order


# TEST EXECUTION ENGINE - Handles code testing and bug detection
class TestExecutionEngine:
    """
    Handles execution of test code and bug detection.
    
    Provides safe execution environment for running generated test cases
    against implementation code, with proper timeout and error handling.
    """
    
    @staticmethod
    async def execute_test_code(test_code: str, timeout: int = 30) -> Tuple[bool, str]:
        """
        Execute test code and return results.
        
        Args:
            test_code: Combined source and test code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Tuple of (has_bugs: bool, test_report: str)
        """
        temp_file = None
        try:
            # Create temporary file for testing
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                delete=False
            ) as f:
                f.write(test_code)
                temp_file = f.name
            
            # Determine appropriate Python executable
            python_cmd = TestExecutionEngine._get_python_command()
            
            # Execute tests with timeout
            process = await asyncio.create_subprocess_exec(
                *python_cmd, '-m', 'unittest', 
                os.path.splitext(os.path.basename(temp_file))[0],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(temp_file)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                stdout_text = stdout.decode() if stdout else ""
                stderr_text = stderr.decode() if stderr else ""
                
                # Determine if tests passed
                has_bugs = process.returncode != 0
                test_report = stdout_text + stderr_text
                
                return has_bugs, test_report
                
            except asyncio.TimeoutError:
                process.kill()
                return True, f"Test execution timeout after {timeout} seconds"
                
        except Exception as e:
            return True, f"Test execution failed: {str(e)}"
            
        finally:
            # Clean up temporary file
            if temp_file:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # Ignore cleanup errors
    
    @staticmethod
    def _get_python_command() -> List[str]:
        """
        Get appropriate Python command for current platform.
        
        Returns:
            List of command components for subprocess execution
        """
        if sys.platform.startswith('win'):
            return [sys.executable]
        else:
            return ['python3']


# MAIN EVOMAC AGENT SYSTEM - Orchestrates the entire multi-agent workflow
class EvoMAC(AgentSystem):
    """
    EvoMAC (Evolutionary Multi-Agent Coding) Agent System.
    
    A sophisticated multi-agent coding framework that simulates a software company
    organization with role-based task decomposition, workflow execution, testing,
    and iterative optimization.
    
    Key Features:
    - Multi-agent task decomposition (CTO role)
    - Distributed code implementation (Programmer roles)
    - Automated testing and quality assurance
    - Iterative bug fixing and optimization
    - Workflow dependency management
    """
    
    def __init__(self, name: str = "evomac", config: Dict[str, Any] = None):
        """
        Initialize EvoMAC system.
        
        Args:
            name: System name identifier
            config: Configuration dictionary containing system parameters
        """
        super().__init__(name, config)
        
        # System configuration
        self.config = config or {}
        self.max_iterations = self.config.get("iteration", 5)
        self.programming_language = self.config.get("language", "python")
        
        # Initialize component managers
        self.code_manager = CodeManager()
        self.test_code_manager = CodeManager()
        self.workflow_organizer = WorkflowOrganizer()
        self.test_workflow_organizer = WorkflowOrganizer()
        
        # Initialize LLM client
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm_client = self._initialize_llm_client()
    
    def _initialize_llm_client(self) -> ChatOpenAI:
        """
        Initialize LangChain OpenAI client with configuration.
        
        Returns:
            Configured ChatOpenAI client
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=0.7
        )
    
    def _format_messages_for_llm(self, system_prompt: str, user_content: str) -> List:
        """
        Format messages for LangChain LLM consumption.
        
        Args:
            system_prompt: System role prompt
            user_content: User message content
            
        Returns:
            List of LangChain message objects
        """
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
    
    async def _call_llm_async(self, messages: List) -> str:
        """
        Call LLM asynchronously and return response.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            LLM response content
        """
        try:
            response = await self.llm_client.ainvoke(
                messages,
                config={"temperature": 0.7}
            )
            return response.content
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
    
    async def _generate_initial_implementation(self, problem_statement: str) -> str:
        """
        Generate initial code implementation.
        
        Args:
            problem_statement: The coding problem to solve
            
        Returns:
            Initial implementation response
        """
        prompt = TASK_PROMPT_TEMPLATES["initial_coding"].format(task=problem_statement)
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["initial_coder"], 
            prompt
        )
        return await self._call_llm_async(messages)
    
    async def _organize_workflow(self, problem_statement: str) -> str:
        """
        Generate workflow organization from CTO agent.
        
        Args:
            problem_statement: The coding problem to solve
            
        Returns:
            Workflow organization response
        """
        format_prompt = self.format_prompt or ""
        prompt = TASK_PROMPT_TEMPLATES["task_organization"].format(
            task=problem_statement,
            language=self.programming_language,
            codes=self.code_manager.get_formatted_codes(),
            format_prompt=format_prompt
        )
        
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["cto"], 
            prompt
        )
        return await self._call_llm_async(messages)
    
    async def _execute_implementation_workflow(self, problem_statement: str) -> None:
        """
        Execute the main implementation workflow based on task decomposition.
        
        Args:
            problem_statement: The coding problem to solve
        """
        composition = self.workflow_organizer.get_composition()
        workflow = self.workflow_organizer.get_workflow()
        
        if not composition or not workflow:
            print("Warning: No valid workflow to execute, using fallback single task")
            # 设置fallback workflow
            self.workflow_organizer._set_fallback_organization()
            composition = self.workflow_organizer.get_composition()
            workflow = self.workflow_organizer.get_workflow()
        
        try:
            # Get execution order using dependency resolution
            execution_order = DependencyResolver.get_execution_order(workflow)
            print(f"Execution order: {execution_order}")
            
            # Execute each task in dependency order
            for task_name in execution_order:
                if task_name in composition:
                    print(f"Executing task: {task_name}")
                    await self._execute_single_task(problem_statement, task_name, composition[task_name])
                else:
                    print(f"Warning: Task {task_name} not found in composition")
                    
        except ValueError as e:
            print(f"Workflow execution failed: {e}")
            # 在失败时，至少执行一个默认任务
            await self._execute_single_task(problem_statement, "Task_1", "Complete the implementation")
    
    async def _execute_single_task(self, problem_statement: str, task_name: str, task_description: str) -> None:
        """
        Execute a single implementation task.
        
        Args:
            problem_statement: The original coding problem
            task_name: Name of the task to execute
            task_description: Description of the task
        """
        format_prompt = self.format_prompt or ""
        prompt = TASK_PROMPT_TEMPLATES["subtask_completion"].format(
            task=problem_statement,
            language=self.programming_language,
            codes=self.code_manager.get_formatted_codes(),
            subtask=task_description,
            format_prompt=format_prompt
        )
        
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["programmer"], 
            prompt
        )
        response = await self._call_llm_async(messages)
        
        # Update implementation with new code
        self.code_manager.update_from_response(response)
    
    async def _execute_testing_workflow(self, problem_statement: str) -> Tuple[bool, str]:
        """
        Execute comprehensive testing workflow.
        
        Args:
            problem_statement: The coding problem to solve
            
        Returns:
            Tuple of (has_bugs: bool, test_reports: str)
        """
        # Generate test organization
        test_organization_response = await self._organize_testing(problem_statement)
        self.test_workflow_organizer.update_from_response(test_organization_response)
        
        # Execute test tasks
        test_composition = self.test_workflow_organizer.get_composition()
        
        if not test_composition:
            return False, "No tests to execute"
        
        # Generate and execute tests for each task
        all_test_reports = []
        has_any_bugs = False
        
        for task_name, test_task_description in test_composition.items():
            has_bugs, test_report = await self._execute_test_task(
                problem_statement, 
                task_name, 
                test_task_description
            )
            
            if has_bugs:
                has_any_bugs = True
            
            all_test_reports.append(f"Test {task_name}: {test_report}")
        
        return has_any_bugs, "\n\n".join(all_test_reports)
    
    async def _organize_testing(self, problem_statement: str) -> str:
        """
        Generate test organization plan.
        
        Args:
            problem_statement: The coding problem to solve
            
        Returns:
            Test organization response
        """
        format_prompt = self.format_prompt or ""
        prompt = TASK_PROMPT_TEMPLATES["test_organization"].format(
            task=problem_statement,
            language=self.programming_language,
            format_prompt=format_prompt
        )
        
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["test_organizer"], 
            prompt
        )
        return await self._call_llm_async(messages)
    
    async def _execute_test_task(self, problem_statement: str, task_name: str, task_description: str) -> Tuple[bool, str]:
        """
        Execute a single test task.
        
        Args:
            problem_statement: The original coding problem
            task_name: Name of the test task
            task_description: Description of the test task
            
        Returns:
            Tuple of (has_bugs: bool, test_report: str)
        """
        format_prompt = self.format_prompt or ""
        test_filename = "test_solution.py"
        
        prompt = TASK_PROMPT_TEMPLATES["test_code_completion"].format(
            task=problem_statement,
            codes=self.code_manager.get_formatted_codes(),
            subtask=task_description,
            test_file_name=test_filename,
            format_prompt=format_prompt
        )
        
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["test_engineer"], 
            prompt
        )
        test_response = await self._call_llm_async(messages)
        
        # Extract and execute test code
        test_code_match = re.search(r'```python\s*(.*?)```', test_response, re.DOTALL)
        if test_code_match:
            test_code = test_code_match.group(1).strip()
            combined_code = self.code_manager.get_raw_code() + "\n\n" + test_code
            return await TestExecutionEngine.execute_test_code(combined_code)
        
        return True, "Failed to extract test code"
    
    async def _perform_iterative_optimization(self, problem_statement: str, test_reports: str) -> Tuple[bool, str]:
        """
        Perform iterative optimization to fix bugs.
        
        Args:
            problem_statement: The coding problem to solve
            test_reports: Current test failure reports
            
        Returns:
            Tuple of (still_has_bugs: bool, final_test_reports: str)
        """
        current_reports = test_reports
        
        for iteration in range(self.max_iterations - 1):
            # Generate update organization
            update_response = await self._organize_updates(problem_statement, current_reports)
            self.workflow_organizer.update_from_response(update_response)
            
            # Execute update workflow
            await self._execute_implementation_workflow(problem_statement)
            
            # Re-test implementation
            has_bugs, new_reports = await self._execute_testing_workflow(problem_statement)
            
            if not has_bugs:
                return False, new_reports
            
            current_reports = new_reports
        
        return True, current_reports
    
    async def _organize_updates(self, problem_statement: str, test_reports: str) -> str:
        """
        Generate update organization to fix current issues.
        
        Args:
            problem_statement: The coding problem to solve
            test_reports: Current test failure reports
            
        Returns:
            Update organization response
        """
        format_prompt = self.format_prompt or ""
        prompt = TASK_PROMPT_TEMPLATES["issue_resolution"].format(
            task=problem_statement,
            codes=self.code_manager.get_formatted_codes(),
            issues=test_reports,
            format_prompt=format_prompt
        )
        
        messages = self._format_messages_for_llm(
            AGENT_SYSTEM_PROMPTS["updater"], 
            prompt
        )
        return await self._call_llm_async(messages)
    
    def _create_message_record(self, content: str, agent_name: str, response_obj: Optional[Any] = None) -> AIMessage:
        """
        Create standardized message record for tracking.
        
        Args:
            content: Message content
            agent_name: Name of the agent that generated the content
            response_obj: Optional response object with metadata
            
        Returns:
            AIMessage with proper metadata
        """
        message = AIMessage(content=content)
        message.name = agent_name
        
        if response_obj and hasattr(response_obj, 'usage_metadata'):
            message.usage_metadata = response_obj.usage_metadata
            
        return message
    
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the complete EvoMAC agent system on a given problem.
        
        This method orchestrates the entire multi-agent workflow:
        1. Initial implementation generation
        2. Task decomposition and workflow organization  
        3. Distributed implementation execution
        4. Comprehensive testing and quality assurance
        5. Iterative optimization and bug fixing
        
        Args:
            problem: Dictionary containing the problem data with 'problem' key
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dictionary containing:
            - messages: List of all agent interactions
            - final_answer: Final implementation code
        """
        problem_statement = problem["problem"]
        conversation_history = []
        
        try:
            # Phase 1: Generate initial implementation
            print("Phase 1: Generating initial implementation...")
            initial_response = await self._generate_initial_implementation(problem_statement)
            self.code_manager.update_from_response(initial_response)
            conversation_history.append(
                self._create_message_record(initial_response, 'initial_coder')
            )
            
            # Phase 2: Organize workflow and task decomposition
            print("Phase 2: Organizing workflow...")
            organization_response = await self._organize_workflow(problem_statement)
            self.workflow_organizer.update_from_response(organization_response)
            conversation_history.append(
                self._create_message_record(organization_response, 'workflow_organizer')
            )
            
            # Phase 3: Execute implementation workflow
            print("Phase 3: Executing implementation workflow...")
            await self._execute_implementation_workflow(problem_statement)
            
            # Phase 4: Execute testing workflow
            print("Phase 4: Executing testing workflow...")
            has_bugs, test_reports = await self._execute_testing_workflow(problem_statement)
            
            # Phase 5: Iterative optimization if bugs found
            if has_bugs:
                print("Phase 5: Performing iterative optimization...")
                has_bugs, test_reports = await self._perform_iterative_optimization(
                    problem_statement, 
                    test_reports
                )
            
            # Get final implementation
            final_implementation = self.code_manager.get_raw_code()
            
            print(f"EvoMAC execution completed. Final implementation has {len(final_implementation)} characters.")
            
            return {
                "messages": conversation_history,
                "final_answer": final_implementation
            }
            
        except Exception as e:
            error_message = self._create_message_record(
                f"EvoMAC execution failed: {str(e)}", 
                'error_handler'
            )
            conversation_history.append(error_message)
            
            print(f"EvoMAC execution failed: {e}")
            
            return {
                "messages": conversation_history,
                "final_answer": f"Error: {str(e)}"
            }


# Register EvoMAC with the agent system registry
AgentSystemRegistry.register(
    "evomac",
    EvoMAC,
    iteration=5,
    language="python"
)

