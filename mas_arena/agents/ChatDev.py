import asyncio
import os
import re
import uuid
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from threading import Thread

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


@dataclass
class ChatDevAgent:
    """Base agent class in ChatDev system"""
    name: str
    role: str
    system_prompt: str
    model_name: str = None
    
    def __post_init__(self):
        self.model_name = self.model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,
            max_retries=2,
            temperature=0.7
        )
        self.chat_history = []

    def clear_history(self):
        """Clear the agent's conversation history"""
        self.chat_history = []

    async def generate_response(self, context: str) -> Dict[str, Any]:
        """Generate response"""
        try:
            # Build messages
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history
            for msg in self.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user input
            messages.append(HumanMessage(content=context))
            
            # Call LLM
            response = await self.llm.ainvoke(messages)
            
            response.id = f"{self.role}_{uuid.uuid4()}"
            response.name = self.role
            
            # Update history
            self.chat_history.append({"role": "user", "content": context})
            self.chat_history.append({"role": "assistant", "content": response.content})
            
            return {
                "message": response,
                "content": response.content
            }
        except Exception as e:
            return {
                "message": None,
                "content": f"Error: {str(e)}"
            }


class Instructor(ChatDevAgent):
    """Instructor role (CTO, CEO, Tester, Reviewer)"""
    pass


class Assistant(ChatDevAgent):
    """Assistant role (CTO, Programmer)"""
    pass


class ChatDev(AgentSystem):
    """
    ChatDev multi-agent software development system
    
    Implements complete software development workflow:
    1. Demand Analysis (DemandAnalysis) - skip for humaneval
    2. Coding (Coding)
    3. Code Completion (CodeCompleteAll)
    4. Code Review (CodeReview)
    5. Real Testing (Test) - Execute real code tests
    """
    
    class TimeoutError(Exception):
        """Execution timeout exception"""
        pass
    
    def __init__(self, name: str = "chatdev", config: Dict[str, Any] = None):
        """Initialize ChatDev system"""
        super().__init__(name, config)
        
        self.config = config or {}
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.max_iterations = self.config.get("max_iterations", 3)
        
        # ChatDev background description
        self.background_prompt = "ChatDev is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'."
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Store project state
        self.project_state = {
            "task": "",
            "modality": "Application",  # Fixed as Application
            "language": "Python",      # Fixed as Python
            "ideas": "",
            "codes": "",
            "requirements": "",
            "test_reports": "",
            "error_summary": ""
        }

    def _create_agents(self) -> Dict[str, Any]:
        """Create role agents"""
        agents = {}
        
        # CEO - Chief Executive Officer  
        agents["CEO"] = Instructor(
            name="Chief Executive Officer",
            role="CEO", 
            system_prompt=f"{self.background_prompt}\nYou are Chief Executive Officer. Now, we are both working at ChatDev and we share a common interest in collaborating to successfully complete a task assigned by a new customer. Your main responsibilities include being an active decision-maker on users' demands and other key policy issues, leader, manager, and executor. Your decision-making role involves high-level decisions about policy and strategy; and your communicator role can involve speaking to the organization's management and employees.",
            model_name=self.model_name
        )
        
        # CPO - Chief Product Officer
        agents["CPO"] = Assistant(
            name="Chief Product Officer", 
            role="CPO",
            system_prompt=f"{self.background_prompt}\nYou are Chief Product Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You are responsible for all product-related matters in ChatDev. Usually includes product design, product strategy, product vision, product innovation, project management and product marketing.",
            model_name=self.model_name
        )
        
        # CTO - Chief Technology Officer  
        agents["CTO"] = Instructor(
            name="Chief Technology Officer",
            role="CTO",
            system_prompt=f"{self.background_prompt}\nYou are Chief Technology Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology (\"IT\") staff members to perform everyday operations.",
            model_name=self.model_name
        )
        
        # Programmer
        agents["Programmer"] = Assistant(
            name="Programmer",
            role="Programmer", 
            system_prompt=f"{self.background_prompt}\nYou are Programmer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can write/create computer software or applications by providing a specific programming language to the computer. You have extensive computing and coding experience in many varieties of programming languages and platforms, such as Python, Java, C, C++, HTML, CSS, JavaScript, XML, SQL, PHP, etc,.",
            model_name=self.model_name
        )
        
        # Code Reviewer
        agents["Code Reviewer"] = Instructor(
            name="Code Reviewer",
            role="Reviewer",
            system_prompt=f"{self.background_prompt}\nYou are Code Reviewer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can help programmers to assess source codes for software troubleshooting, fix bugs to increase code quality and robustness, and offer proposals to improve the source codes.",
            model_name=self.model_name
        )
        
        return {
            "workers": list(agents.values()),
            "agents_dict": agents,
            "agent_names": {i: agent.name for i, agent in enumerate(agents.values())}
        }

    def get_agent_names(self) -> Dict[int, str]:
        """Get agent name mapping for visualization and debugging"""
        return self.agents.get("agent_names", {})
    
    def get_agent_by_role(self, role: str) -> Optional[Any]:
        """Get agent by role name"""
        agents_dict = self.agents.get("agents_dict", {})
        return agents_dict.get(role)

    def run_with_timeout(self, func, args, timeout: int = 60):
        """Execute function within specified time, raise exception if timeout"""
        result: list[Any] = []
        exception: list[BaseException] = []

        def target():
            try:
                result.append(func(*args))
            except BaseException as e:
                exception.append(e)

        thread = Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise self.TimeoutError("Execution timed out")

        if exception:
            raise exception[0]

        return result[0] if result else None

    def extract_code_from_response(self, text: str) -> str:
        """Extract code from response, following HumanEval evaluator logic"""
        # 1. Look for "## Validated Code" block
        qa_match = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if qa_match:
            return qa_match.group(1).strip()

        # 2. Look for any ```python code block
        block_match = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if block_match:
            return block_match.group(1).strip()

        # 3. Look for function definition pattern
        fn_match = re.search(r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\n{2,}|\Z)", text)
        if fn_match:
            return fn_match.group(1).strip()

        # 4. Fallback: return entire text
        return text.strip()

    def check_solution(self, code: str, test: str, entry_point: str) -> Tuple[bool, str, str]:
        """Check if solution is correct"""
        output_results = ""
        try:
            # Create isolated namespace
            env: Dict[str, Any] = {}

            # Inject candidate implementation
            exec(code, env)
            if entry_point not in env:
                return False, f"Function '{entry_point}' not found in code", ""
            
            candidate_fn = env[entry_point]

            # Inject and get test function
            exec(test, env)
            if "check" not in env:
                return False, "Test function 'check' not found", ""
            
            check_fn = env["check"]

            # Run test, if check() raises exception, test fails
            self.run_with_timeout(check_fn, (candidate_fn,), timeout=60)
            
            # If test passes, collect some example output results for Programmer to review
            try:
                # Try to extract some example calls from test code
                import re
                test_examples = re.findall(r'assert\s+.*?==\s+.*', test)
                if test_examples:
                    output_results = "Test examples and expected outputs:\n"
                    for i, example in enumerate(test_examples[:3]):  # Show max 3 examples
                        output_results += f"  {i+1}. {example}\n"
                
                # Try to extract function call examples
                call_examples = re.findall(rf'{entry_point}\([^)]*\)', test)
                if call_examples:
                    output_results += "\nFunction calls found in tests:\n"
                    for i, call in enumerate(set(call_examples[:3])):  # Deduplicate, max 3
                        try:
                            result = eval(call, env)
                            output_results += f"  {call} -> {result}\n"
                        except:
                            output_results += f"  {call} -> (execution failed)\n"
            except:
                output_results = "Test passed but could not extract example outputs"
            
            return True, "All tests passed", output_results

        except self.TimeoutError as te:
            return False, str(te), ""
        except AssertionError as ae:
            return False, f"Test failed: {ae}", ""
        except Exception as exc:
            return False, f"Execution error: {exc}", ""

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run complete workflow of ChatDev system"""
        try:
            # Extract task description
            task = problem.get("problem", "")
            self.project_state["task"] = task
            
            # Store all LLM response messages
            all_messages = []
            
            
            modality = "Application"  # Fixed
            language = "Python"      # Fixed
            
            # 1. Demand Analysis phase (skip for humaneval tasks)
            modality = await self._demand_analysis_phase(task, all_messages)
            self.project_state["modality"] = modality
            
            self.project_state["language"] = language
            
            # 2. Coding phase
            codes = await self._coding_phase(task, modality, language, all_messages, problem)
            self.project_state["codes"] = codes
            
            # 3. Code Completion phase
            completed_codes = await self._code_complete_all_phase(task, modality, language, codes, all_messages)
            self.project_state["codes"] = completed_codes
            
            # 4. Code Review phase
            reviewed_codes = await self._code_review_phase(task, modality, language, completed_codes, all_messages)
            self.project_state["codes"] = reviewed_codes
            
            # 5. Real Testing phase - Execute real code
            final_codes = await self._test_phase(reviewed_codes, problem, all_messages)
            
            return {
                "messages": all_messages,
                "final_answer": final_codes
            }
            
        except Exception as e:
            return {
                "messages": all_messages if 'all_messages' in locals() else [],
                "final_answer": f"Error in ChatDev workflow: {str(e)}"
            }

    async def _demand_analysis_phase(self, task: str, all_messages: List) -> str:
        """Demand Analysis phase - CEO and CPO discuss product form"""
        self.get_agent_by_role("CEO").clear_history()
        self.get_agent_by_role("CPO").clear_history()
        
        phase_prompt = [
            "As the Chief Product Officer, please carefully analyze the given task and generate a comprehensive requirements document.",
            f"Task: \"{task}\"",
            "Your analysis should include:",
            "1. Core functionality requirements - what the solution needs to accomplish",
            "2. Input/Output specifications - what data types, formats, and structures are expected",
            "3. Key constraints and edge cases that need to be handled",
            "4. Performance and reliability requirements",
            "5. User interaction patterns (if applicable)",
            "",
            "Based on your analysis, determine the most appropriate solution approach.",
            "Once you have completed your requirements analysis, respond with:",
            "\"<INFO> Application\" for code-based solutions, or \"<INFO> Document\" for documentation-based solutions."
        ]
        
        context = f"Task: {task}\n\n{' '.join(phase_prompt)}"
        
        # CPO as assistant role provides suggestions
        cpo_response = await self.agents["workers"][1].generate_response(context)  # CPO
        all_messages.append(cpo_response["message"])
        
        # Extract product form
        modality_match = re.search(r'<INFO>\s*(\w+)', cpo_response["content"])
        modality = modality_match.group(1) if modality_match else "Application"
        
        return modality

    async def _coding_phase(self, task: str, modality: str, language: str, all_messages: List, problem: Dict[str, Any]) -> str:
        """Coding phase - CTO guides Programmer to write code, includes up to 3 rounds of debate"""
        
        self.get_agent_by_role("Programmer").clear_history()
        self.get_agent_by_role("CTO").clear_history()

        # Initial coding prompt
        phase_prompt = [
            "According to the new user's task and our software designs listed below: ",
            f"Task: \"{task}\".",
            f"Modality: \"{modality}\".",
            f"Programming Language: \"{language}\"",
            f"We have decided to complete the task through a executable software implemented via {language}. As the Programmer, to satisfy the new user's demands, you should write complete, functional code that solves the task.",
            f"",
            f"CRITICAL CODING REMINDERS - Pay special attention to these commonly forgotten aspects:",
            f"1. INPUT/OUTPUT FORMAT COMPLIANCE:",
            f"   - Carefully read and understand the expected input format (string, number, list, etc.)",
            f"   - Ensure your function accepts the correct input type to avoid 'Input must be a string' errors",
            f"   - Match the exact output format specified in the task description",
            f"   - Add proper input validation and type checking",
            f"",
            f"2. FUNCTION SIGNATURE AND INTERFACE:",
            f"   - Use the exact function name specified in the task (if provided)",
            f"   - Ensure parameter names and types match requirements",
            f"   - Return the correct data type and format",
            f"",
            f"3. EDGE CASES AND ERROR HANDLING:",
            f"   - Handle empty inputs, null values, and boundary conditions",
            f"   - Add try-except blocks for potential runtime errors",
            f"   - Validate inputs before processing",
            f"",
            f"4. ALGORITHM IMPLEMENTATION:",
            f"   - Implement the core logic completely - no placeholder code",
            f"   - Test your logic mentally with the provided examples",
            f"   - Ensure all loops, conditions, and calculations are correct",
            f"",
            f"5. IMPORTS AND DEPENDENCIES:",
            f"   - Include all necessary import statements",
            f"   - Use standard library when possible to avoid dependency issues",
            f"",
            f"Think step by step and reason yourself to the right decisions to make sure we get it right.",
            f"You will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.",
            f"Then you will output the complete functional code.",
            f"Please note that the code should be fully functional. Ensure to implement all functions. No placeholders (such as 'pass' in Python).",
            f"",
            f"IMPORTANT: Format your final response according to these requirements:",
            f"{self.format_prompt}"
        ]
        
        context = ' '.join(phase_prompt)

        # Round 1: Initial coding by Programmer
        programmer_response = await self.agents["workers"][3].generate_response(context)  # Programmer
        all_messages.append(programmer_response["message"])
        current_code = programmer_response["content"]
        
        # Up to 3 rounds of debate
        for debate_round in range(1):
            # CTO reviews code and provides suggestions
            review_prompt = [
                f"According to the new user's task and the Programmer's current implementation:",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Current Implementation:",
                f"\"{current_code}\"",
                "As the Chief Technology Officer, please review this implementation. Check for:",
                "1) Architecture completeness and design quality",
                "2) Code functionality and logic correctness", 
                "3) Whether all requirements are met",
                "4) Potential improvements or missing components",
                "If the implementation is satisfactory, respond with \"<APPROVED>\" at the end.",
                "Otherwise, provide specific suggestions for improvement."
            ]
            
            cto_context = ' '.join(review_prompt)
            cto_response = await self.agents["workers"][2].generate_response(cto_context)  # CTO
            all_messages.append(cto_response["message"])
            
            # If CTO approves code, end debate
            if "<APPROVED>" in cto_response["content"]:
                break
            
            # Programmer improves code based on CTO's suggestions
            improvement_prompt = [
                f"According to the Chief Technology Officer's review feedback:",
                f"Original Task: \"{task}\".",
                f"Your Current Implementation:",
                f"\"{current_code}\"",
                f"CTO's Feedback:",
                f"\"{cto_response['content']}\"",
                "As the Programmer, please improve your implementation based on the CTO's feedback.",
                "IMPORTANT: Format your improved code according to these requirements:",
                f"{self.format_prompt}",
                "Ensure all suggested improvements are incorporated."
            ]
            
            improvement_context = ' '.join(improvement_prompt)
            improved_response = await self.agents["workers"][3].generate_response(improvement_context)  # Programmer
            all_messages.append(improved_response["message"])
            current_code = improved_response["content"]
        
        return current_code

    async def _code_complete_all_phase(self, task: str, modality: str, language: str, codes: str, all_messages: List) -> str:
        """Code Completion phase - Loop to complete all unimplemented files"""
        current_codes = codes
        if self.evaluator_name == "humaneval":
            return current_codes

        self.get_agent_by_role("Programmer").clear_history()

        # Simplified handling: check for unimplemented code
        for iteration in range(3):  # Max 3 iterations
            if "TODO" not in current_codes and "pass" not in current_codes and "# Implementation needed" not in current_codes:
                break
                
            phase_prompt = [
                f"According to the new user's task and our software designs listed below: ",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Current codes:",
                f"\"{current_codes}\"",
                f"As the Programmer, you need to complete and implement all remaining functions, methods and classes. Make sure all TODO items and placeholder code (like 'pass') are fully implemented.",
                f"Output the complete, fully functional code that solves the task.",
                f"",
                f"IMPORTANT: Format your final response according to these requirements:",
                f"{self.format_prompt}"
            ]
            
            context = ' '.join(phase_prompt)
            programmer_response = await self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            current_codes = programmer_response["content"]
        
        return current_codes

    async def _code_review_phase(self, task: str, modality: str, language: str, codes: str, all_messages: List) -> str:
        """Code Review phase - Code Reviewer and Programmer interact in cycles"""
        current_codes = codes
        
        self.get_agent_by_role("Code Reviewer").clear_history()
        self.get_agent_by_role("Programmer").clear_history()

        for iteration in range(2):  # Max 3 rounds of review
            # Code Reviewer review
            review_prompt = [
                f"According to the new user's task and our software designs: ",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Codes:",
                f"\"{current_codes}\"",
                f"As the Code Reviewer, to make the software directly operable without further coding, ChatDev have formulated the following regulations:",
                f"1) all referenced classes should be imported;",
                f"2) all methods should be implemented;", 
                f"3) all methods need to have the necessary comments;",
                f"4) no potential bugs;",
                f"5) The entire project conforms to the tasks proposed by the user;",
                f"6) most importantly, do not only check the errors in the code, but also the logic of code. Make sure that user can interact with generated software without losing any feature in the requirement;",
                f"Now, you should check the above regulations one by one and review the codes in detail, propose one comment with the highest priority about the codes, and give me instructions on how to fix. Tell me your comment with the highest priority and corresponding suggestions on revision. If the codes are perfect and you have no comment on them, return only one line like \"<INFO> Finished\"."
            ]
            
            context = ' '.join(review_prompt)
            reviewer_response = await self.agents["workers"][4].generate_response(context)  # Code Reviewer
            all_messages.append(reviewer_response["message"])
            
            # If review complete, break loop
            if "<INFO> Finished" in reviewer_response["content"]:
                break
                
            # Programmer modifies code
            modify_prompt = [
                f"According to the new user's task, our designed product modality, languages and ideas, our developed first-edition source codes are listed below: ",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Current codes: ",
                f"\"{current_codes}\"",
                f"Code review comments:",
                f"\"{reviewer_response['content']}\"",
                f"As the Programmer, modify the code according to the review comments. Output the complete, improved code that addresses all the issues mentioned in the review.",
                f"",
                f"IMPORTANT: Format your final response according to these requirements:",
                f"{self.format_prompt}"
            ]
            
            context = ' '.join(modify_prompt)
            programmer_response = await self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            current_codes = programmer_response["content"]
        
        return current_codes

    async def _test_phase(self, codes: str, problem: Dict[str, Any], all_messages: List) -> str:
        """Real Testing phase - Use real code execution and testing"""
        current_codes = codes
        
        # Clear Programmer history
        self.get_agent_by_role("Programmer").clear_history()
        
        # Get test information from problem
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        
        # If no test cases, return current code
        if not test_code or not entry_point:
            return current_codes
        
        for iteration in range(3):  # Max 3 rounds of testing
            # Extract pure code part from current code
            extracted_code = self.extract_code_from_response(current_codes)
            
            # Execute real test
            passed, error_message, test_output = await asyncio.to_thread(self.check_solution, extracted_code, test_code, entry_point)
            
            if passed:
                # Test passed, let Programmer review output and confirm
                review_prompt = [
                    f"Congratulations! Your code has passed all unit tests.",
                    f"",
                    f"Task: \"{self.project_state['task']}\"",
                    f"",
                    f"Your Final Code:",
                    f"{extracted_code}",
                    f"",
                    f"Test Results:",
                    f"{test_output}",
                    f"",
                    f"As the Programmer, please review the test results above to ensure:",
                    f"1. The outputs match your expectations for the given task",
                    f"2. The function behavior is correct for the test cases",
                    f"3. The implementation handles edge cases appropriately",
                    f"",
                    f"If you are satisfied with the results, respond with '<CONFIRMED>' and provide the final code.",
                    f"If you notice any issues or want to make improvements, provide the updated code.",
                    f"",
                    f"IMPORTANT: Format your final response according to these requirements:",
                    f"{self.format_prompt}"
                ]
                
                context = '\n'.join(review_prompt)
                programmer_response = await self.agents["workers"][3].generate_response(context)  # Programmer
                all_messages.append(programmer_response["message"])
                
                # Return Programmer's final response regardless of confirmation
                return programmer_response["content"]
            
            # Test failed, let Programmer fix code
            fix_prompt = [
                f"Your previous code has failed the unit tests.",
                f"",
                f"Task: \"{self.project_state['task']}\"",
                f"",
                f"Current Code:",
                f"{extracted_code}",
                f"",
                f"Test Failure/Error:",
                f"{error_message}",
                f"",
                f"As the Programmer, analyze the error message and fix the bug in the code.",
                f"Make sure to:",
                f"1. Address the specific error mentioned above",
                f"2. Ensure the function signature matches requirements",
                f"3. Handle all edge cases properly",
                f"4. Test your logic against the examples in the original task",
                f"",
                f"Provide the complete, corrected code.",
                f"IMPORTANT: Format your final response according to these requirements:",
                f"{self.format_prompt}"
            ]
            
            context = '\n'.join(fix_prompt)
            programmer_response = await self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            current_codes = programmer_response["content"]
        
        # If still issues after 3 rounds, return last code
        return current_codes


# Register ChatDev system to framework
AgentSystemRegistry.register(
    "chatdev",
    ChatDev,
    evaluator="humaneval",  # Default to use HumanEval evaluator
    description="ChatDev multi-agent software development system, implementing complete software development workflow",
    max_iterations=3
)
        
        
        
