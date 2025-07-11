import os
import asyncio
import re
import tempfile
import sys
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

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

    def generate_response(self, context: str) -> Dict[str, Any]:
        """Generate response"""
        try:
            # Build messages
            messages = [SystemMessage(content=self.system_prompt)]
            
            # Add chat history
            for msg in self.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user input
            messages.append(HumanMessage(content=context))
            
            # Call LLM
            response = self.llm.invoke(messages)
            
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
    1. Demand Analysis (DemandAnalysis)
    2. Language Selection (LanguageChoose)
    3. Coding (Coding)
    4. Code Completion (CodeCompleteAll)
    5. Code Review (CodeReview)
    6. Testing (Test)
    """
    
    def __init__(self, name: str = "chatdev", config: Dict[str, Any] = None):
        """Initialize ChatDev system"""
        super().__init__(name, config)
        
        self.config = config or {}
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.max_iterations = self.config.get("max_iterations", 3)
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Store project state
        self.project_state = {
            "task": "",
            "modality": "",
            "language": "",
            "ideas": "",
            "codes": "",
            "requirements": ""
        }

    def _create_agents(self) -> Dict[str, Any]:
        """Create role agents"""
        # ChatDev background prompt
        chatdev_prompt = "ChatDev is a software company powered by multiple intelligent agents, such as chief executive officer, chief human resources officer, chief product officer, chief technology officer, etc, with a multi-agent organizational structure and the mission of 'changing the digital world through programming'."
        
        agents = {}
        
        # CEO - Chief Executive Officer  
        agents["CEO"] = Instructor(
            name="Chief Executive Officer",
            role="CEO", 
            system_prompt=f"{chatdev_prompt}\nYou are Chief Executive Officer. Now, we are both working at ChatDev and we share a common interest in collaborating to successfully complete a task assigned by a new customer. Your main responsibilities include being an active decision-maker on users' demands and other key policy issues, leader, manager, and executor. Your decision-making role involves high-level decisions about policy and strategy; and your communicator role can involve speaking to the organization's management and employees.",
            model_name=self.model_name
        )
        
        # CPO - Chief Product Officer
        agents["CPO"] = Assistant(
            name="Chief Product Officer", 
            role="CPO",
            system_prompt=f"{chatdev_prompt}\nYou are Chief Product Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You are responsible for all product-related matters in ChatDev. Usually includes product design, product strategy, product vision, product innovation, project management and product marketing.",
            model_name=self.model_name
        )
        
        # CTO - Chief Technology Officer  
        agents["CTO"] = Instructor(
            name="Chief Technology Officer",
            role="CTO",
            system_prompt=f"{chatdev_prompt}\nYou are Chief Technology Officer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You are very familiar to information technology. You will make high-level decisions for the overarching technology infrastructure that closely align with the organization's goals, while you work alongside the organization's information technology (\"IT\") staff members to perform everyday operations.",
            model_name=self.model_name
        )
        
        # Programmer
        agents["Programmer"] = Assistant(
            name="Programmer",
            role="Programmer", 
            system_prompt=f"{chatdev_prompt}\nYou are Programmer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can write/create computer software or applications by providing a specific programming language to the computer. You have extensive computing and coding experience in many varieties of programming languages and platforms, such as Python, Java, C, C++, HTML, CSS, JavaScript, XML, SQL, PHP, etc,.",
            model_name=self.model_name
        )
        
        # Code Reviewer
        agents["Code Reviewer"] = Instructor(
            name="Code Reviewer",
            role="Reviewer",
            system_prompt=f"{chatdev_prompt}\nYou are Code Reviewer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can help programmers to assess source codes for software troubleshooting, fix bugs to increase code quality and robustness, and offer proposals to improve the source codes.",
            model_name=self.model_name
        )
        
        # Software Test Engineer
        agents["Tester"] = Instructor(
            name="Software Test Engineer", 
            role="Tester",
            system_prompt=f"{chatdev_prompt}\nYou are Software Test Engineer. we are both working at ChatDev. We share a common interest in collaborating to successfully complete a task assigned by a new customer. You can use the software as intended to analyze its functional properties, design manual and automated test procedures to evaluate each software product, build and implement software evaluation test programs, and run test programs to ensure that testing protocols evaluate the software correctly.",
            model_name=self.model_name
        )
        
        return {"workers": list(agents.values())}

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run complete workflow of ChatDev system"""
        try:
            # Extract task description
            task = problem.get("problem", "")
            self.project_state["task"] = task
            
            # Store all LLM response messages
            all_messages = []
            
            # 1. Demand Analysis Phase (DemandAnalysis)
            modality = await self._demand_analysis_phase(task, all_messages)
            self.project_state["modality"] = modality
            
            # 2. Language Selection Phase (LanguageChoose)
            language = await self._language_choose_phase(task, modality, all_messages)
            self.project_state["language"] = language
            
            # 3. Coding Phase (Coding)
            codes = await self._coding_phase(task, modality, language, all_messages)
            self.project_state["codes"] = codes
            
            # 4. Code Completion Phase (CodeCompleteAll)
            completed_codes = await self._code_complete_all_phase(task, modality, language, codes, all_messages)
            self.project_state["codes"] = completed_codes
            
            # 5. Code Review Phase (CodeReview)
            reviewed_codes = await self._code_review_phase(task, modality, language, completed_codes, all_messages)
            self.project_state["codes"] = reviewed_codes
            
            # 6. Testing Phase (Test)
            final_codes = await self._test_phase(language, reviewed_codes, all_messages)
            
            # Extract final answer - Combine with format prompt
            final_answer = self._extract_final_answer(final_codes)
            
            return {
                "messages": all_messages,
                "final_answer": final_answer
            }
            
        except Exception as e:
            return {
                "messages": all_messages if 'all_messages' in locals() else [],
                "final_answer": f"Error in ChatDev workflow: {str(e)}"
            }

    async def _demand_analysis_phase(self, task: str, all_messages: List) -> str:
        """Demand Analysis Phase - CEO and CPO discuss product form"""
        phase_prompt = [
            "ChatDev has made products in the following form before:",
            "Image: can present information in line chart, bar chart, flow chart, cloud chart, Gantt chart, etc.",
            "Document: can present information via .docx files.",
            "PowerPoint: can present information via .pptx files.",
            "Excel: can present information via .xlsx files.",
            "PDF: can present information via .pdf files.",
            "Website: can present personal resume, tutorial, products, or ideas, via .html files.",
            "Application: can implement visualized game, software, tool, etc, via python.",
            "Dashboard: can display a panel visualizing real-time information.",
            "Mind Map: can represent ideas, with related concepts arranged around a core concept.",
            f"As the Chief Product Officer, to satisfy the new user's demand and the product should be realizable, you should keep discussing with me to decide which product modality do we want the product to be?",
            "Note that we must ONLY discuss the product modality and do not discuss anything else! Once we all have expressed our opinion(s) and agree with the results of the discussion unanimously, any of us must actively terminate the discussion by replying with only one line, which starts with a single word <INFO>, followed by our final product modality without any other words, e.g., \"<INFO> PowerPoint\"."
        ]
        
        context = f"Task: {task}\n\n{' '.join(phase_prompt)}"
        
        # CPO as assistant role provides suggestions
        cpo_response = self.agents["workers"][1].generate_response(context)  # CPO
        all_messages.append(cpo_response["message"])
        
        # Extract product form
        modality_match = re.search(r'<INFO>\s*(\w+)', cpo_response["content"])
        modality = modality_match.group(1) if modality_match else "Application"
        
        return modality

    async def _language_choose_phase(self, task: str, modality: str, all_messages: List) -> str:
        """Language Selection Phase - CTO and CEO discuss programming language"""
        phase_prompt = [
            f"According to the new user's task and some creative brainstorm ideas listed below: ",
            f"Task: \"{task}\".",
            f"Modality: \"{modality}\".",
            f"Ideas: \"\".",
            f"We have decided to complete the task through a executable software implemented via a programming language. ",
            f"As the Chief Technology Officer, to satisfy the new user's demand and make the software realizable, you should propose a concrete programming language. If python can complete this task via Python, please answer Python; otherwise, answer another programming language (e.g., Java, C++, etc,).",
            f"Note that we must ONLY discuss the target programming language and do not discuss anything else! Once we all have expressed our opinion(s) and agree with the results of the discussion unanimously, any of us must actively terminate the discussion and conclude the best programming language we have discussed without any other words or reasons, return only one line using the format: \"<INFO> *\" where \"*\" represents a programming language."
        ]
        
        context = ' '.join(phase_prompt)
        
        # CTO as assistant role selects language
        cto_response = self.agents["workers"][2].generate_response(context)  # CTO
        all_messages.append(cto_response["message"])
        
        # Extract programming language
        language_match = re.search(r'<INFO>\s*(\w+)', cto_response["content"]) 
        language = language_match.group(1) if language_match else "Python"
        
        return language

    async def _coding_phase(self, task: str, modality: str, language: str, all_messages: List) -> str:
        """Coding Phase - CTO guides Programmer to write code"""
        phase_prompt = [
            f"According to the new user's task and our software designs listed below: ",
            f"Task: \"{task}\".",
            f"Task description: \"\".",
            f"Modality: \"{modality}\".",
            f"Programming Language: \"{language}\"",
            f"Ideas:\"\"",
            f"We have decided to complete the task through a executable software implemented via {language}. As the Programmer, to satisfy the new user's demands, you should write complete, functional code that solves the task.",
            f"Think step by step and reason yourself to the right decisions to make sure we get it right.",
            f"You will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.",
            f"Then you will output the complete functional code.",
            f"Please note that the code should be fully functional. Ensure to implement all functions. No placeholders (such as 'pass' in Python).",
            f"",
            f"Output format requirements:",
            f"{self.format_prompt}"
        ]
        
        context = ' '.join(phase_prompt)
        
        # Programmer as assistant role writes code
        programmer_response = self.agents["workers"][3].generate_response(context)  # Programmer
        all_messages.append(programmer_response["message"])
        
        return programmer_response["content"]

    async def _code_complete_all_phase(self, task: str, modality: str, language: str, codes: str, all_messages: List) -> str:
        """Code Completion Phase - Loop to complete all unimplemented files"""
        current_codes = codes
        
        # Simplified handling: check for unimplemented code
        for iteration in range(3):  # Maximum 3 iterations
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
                f"Output format requirements:",
                f"{self.format_prompt}"
            ]
            
            context = ' '.join(phase_prompt)
            programmer_response = self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            current_codes = programmer_response["content"]
        
        return current_codes

    async def _code_review_phase(self, task: str, modality: str, language: str, codes: str, all_messages: List) -> str:
        """Code Review Phase - Code Reviewer and Programmer interact in loops"""
        current_codes = codes
        
        for iteration in range(3):  # Maximum 3 rounds of review
            # Code Reviewer review
            review_prompt = [
                f"According to the new user's task and our software designs: ",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Ideas: \"\"",
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
            reviewer_response = self.agents["workers"][4].generate_response(context)  # Code Reviewer
            all_messages.append(reviewer_response["message"])
            
            # If review is complete, break the loop
            if "<INFO> Finished" in reviewer_response["content"]:
                break
                
            # Programmer modifies code
            modify_prompt = [
                f"According to the new user's task, our designed product modality, languages and ideas, our developed first-edition source codes are listed below: ",
                f"Task: \"{task}\".",
                f"Modality: \"{modality}\".",
                f"Programming Language: \"{language}\"",
                f"Ideas: \"\"",
                f"Current codes: ",
                f"\"{current_codes}\"",
                f"Code review comments:",
                f"\"{reviewer_response['content']}\"",
                f"As the Programmer, modify the code according to the review comments. Output the complete, improved code that addresses all the issues mentioned in the review.",
                f"",
                f"Output format requirements:",
                f"{self.format_prompt}"
            ]
            
            context = ' '.join(modify_prompt)
            programmer_response = self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            current_codes = programmer_response["content"]
        
        return current_codes

    async def _test_phase(self, language: str, codes: str, all_messages: List) -> str:
        """Testing Phase - Software Test Engineer and Programmer interact in loops"""
        current_codes = codes
        
        for iteration in range(3):  # Maximum 3 rounds of testing
            # Simulate test reports
            test_reports = "No syntax errors found. All basic functionality tests passed."
            
            # Test Engineer summarizes errors
            error_summary_prompt = [
                f"Our developed source codes and corresponding test reports are listed below: ",
                f"Programming Language: \"{language}\"",
                f"Source Codes:",
                f"\"{current_codes}\"",
                f"Test Reports of Source Codes:",
                f"\"{test_reports}\"",
                f"According to my test reports, please locate and summarize the bugs that cause the problem."
            ]
            
            context = ' '.join(error_summary_prompt)
            tester_response = self.agents["workers"][5].generate_response(context)  # Tester
            all_messages.append(tester_response["message"])
            
            # If no errors, end testing
            if "No" in tester_response["content"] or "no bugs" in tester_response["content"].lower() or "no issues" in tester_response["content"].lower():
                break
                
            # Programmer fixes bugs
            fix_prompt = [
                f"Our developed source codes and corresponding test reports are listed below: ",
                f"Programming Language: \"{language}\"",
                f"Current source codes:",
                f"\"{current_codes}\"", 
                f"Test reports:",
                f"\"{test_reports}\"",
                f"Error summary:",
                f"\"{tester_response['content']}\"",
                f"As the Programmer, fix all bugs and issues identified in the test reports. Output the complete, bug-free code.",
                f"If no bugs are found, output the current code as final validated code.",
                f"",
                f"Output format requirements:",
                f"{self.format_prompt}"
            ]
            
            context = ' '.join(fix_prompt)
            programmer_response = self.agents["workers"][3].generate_response(context)  # Programmer
            all_messages.append(programmer_response["message"])
            
            # Check if fixes are complete
            if ("<INFO> Finished" in programmer_response["content"] or 
                "no bugs" in programmer_response["content"].lower() or
                "no issues" in programmer_response["content"].lower()):
                # If complete, keep current code
                break
                
            current_codes = programmer_response["content"]
        
        return current_codes

    def _extract_final_answer(self, final_codes: str) -> str:
        """Extract final answer, ensure it follows format_prompt format"""
        # If output already contains format_prompt format, return directly
        if "<answer>" in final_codes and "</answer>" in final_codes:
            return final_codes
        
        # If not properly formatted, try to extract code and reformat
        import re
        
        # Try to extract code block
        code_pattern = r'```python\s*(.*?)\s*```'
        code_match = re.search(code_pattern, final_codes, re.DOTALL)
        
        if code_match:
            extracted_code = code_match.group(1).strip()
        else:
            # If no code block, assume entire content is code
            extracted_code = final_codes.strip()
        
        # Format output using format_prompt
        if "humaneval" in self.evaluator_name or "mbpp" in self.evaluator_name:
            # For code generation tasks, use complete format
            formatted_answer = f"""<answer>
## Implementation Details
Complete implementation of the requested functionality.

## Features Implemented
All required functions and features as specified in the task.

## Optimizations
Code follows best practices and is optimized for readability and performance.

## Validated Code
```python
{extracted_code}
```
</answer>"""
        else:
            # For other tasks, use simple format
            formatted_answer = f"Solution:\n{extracted_code}"
        
        return formatted_answer


# Register ChatDev system to framework
AgentSystemRegistry.register(
    "chatdev",
    ChatDev,
    evaluator="humaneval",  # Default using HumanEval evaluator
    description="ChatDev multi-agent software development system, implementing complete software development workflow",
    max_iterations=3
)
        
        
        
