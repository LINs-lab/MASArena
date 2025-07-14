"""
ADAS (Agentic Decision-Making and Self-Correction) Framework Implementation

This module implements a general ADAS system that integrates into the MASArena framework.
The ADAS framework uses a "try-reflect-retry" iterative process for systematic solution exploration.
"""

import os
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import namedtuple


import openai
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry


# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])


class ADASPrompts:
    """ADAS Prompt templates based on prompt.jsonl"""
    
    SYSTEM_PROMPT = """You are a helpful assistant. You MUST return your response as a WELL-FORMED JSON object with the following strict requirements:

1. Use ONLY double quotes (") for strings, never single quotes (')
2. ALL property names MUST be enclosed in double quotes
3. No trailing commas
4. No unescaped newlines or special characters in string values
5. Use proper JSON escape sequences (\\n for newlines, \\" for quotes, \\\\ for backslashes)
6. Ensure the JSON is complete and parseable

Example of correct format:
{"thought": "This is my reasoning...", "name": "Agent Name", "code": "def function():\\n    return result"}

CRITICAL: Your response must be valid JSON that can be parsed by json.loads() in Python."""
    
    MAIN_PROMPT = """You are an expert machine learning researcher testing various agentic systems. 
Your objective is to design building blocks such as prompts and workflows within these systems to solve complex tasks. 
Your aim is to design an optimal agent performing well on the given domain.

# Your task 
You are deeply familiar with prompting techniques and the agent works from the literature. 
Your goal is to maximize the specified performance metrics by proposing interestingly new agents. Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them. 
Be creative when thinking about the next interesting agent to try. 
You are encouraged to draw inspiration from related agent papers or academic papers from other research areas. 
Use the knowledge from the archive and inspiration from academic literature to propose the next interesting agentic system design. 
THINK OUTSIDE THE BOX.

# Output Instruction and Example: 
The first key should be ("thought"), and it should capture your thought process for designing the next function. 
In the "thought" section, first reason about what the next interesting agent to try should be, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps. 
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact "forward()" function in Python code that you would like to try. 
You must write COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.  
Here is an example of the output format for the next agent (MUST be valid JSON with double quotes): 
{"thought": "**Insights:** Your insights on what should be the next interesting agent. **Overall Idea:** your reasoning and the overall concept behind the agent design. **Implementation:** describe the implementation step by step.", "name": "Name of your proposed agent", "code": "def forward(self, taskInfo):\\n    # Your code here\\n    return result"}

CRITICAL: Your response MUST be a valid JSON object using double quotes for all strings and property names. Escape all special characters properly (use \\n for newlines, \\" for quotes, \\\\ for backslashes)."""

    REFLECTION_PROMPT_R1 = """Carefully review the proposed new architecture and reflect on the following points:  
1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts. 
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation. 
- Decide whether the current architecture is innovative. 
- USE CRITICAL THINKING!  

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting. 
- Observe carefully about whether the implementation is actually doing what it is supposed to do. 
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation. 
- Try to avoid the implementation being too similar to the previous agent.  

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.  

Your response should be organized as follows:  
"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements. 
"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response. 
"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.) 
"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.

CRITICAL: Your response MUST be a valid JSON object using double quotes for all strings and property names. Escape all special characters properly (use \\n for newlines, \\" for quotes, \\\\ for backslashes)."""

    REFLECTION_PROMPT_R2 = """Further revise the code. Your response should be organized as follows: 
Include your updated reflections in the "reflection". 
Repeat the previous "thought" and "name". 
Update the corrected version of the code in the "code" section.

CRITICAL: Your response MUST be a valid JSON object using double quotes for all strings and property names. Escape all special characters properly (use \\n for newlines, \\" for quotes, \\\\ for backslashes)."""

    ERROR_REFLECTION_PROMPT = """Error during evaluation: {error_msg}
Carefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. 
Repeat your previous thought in "thought", and put your thinking for debugging in "debug thought".

CRITICAL: Your response MUST be a valid JSON object using double quotes for all strings and property names. Escape all special characters properly (use \\n for newlines, \\" for quotes, \\\\ for backslashes)."""


@dataclass  
class Agent:
    """Represents an LLM agent with memory and reflection capabilities"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str = ""
    chat_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,
            max_retries=2,
            temperature=0.7
        )

    def generate_response(self, prompt: str, iteration_idx: int = 0) -> Dict[str, Any]:
        """Generate response from the agent"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Simple retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                
                # Ensure usage_metadata is in the correct format to avoid KeyError in base.py
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    # If it's a CompletionUsage object, it should be handled correctly by base.py
                    # If it's a dict, make sure it has all required fields
                    usage_metadata = response.usage_metadata
                    if isinstance(usage_metadata, dict):
                        # Ensure all required fields are present
                        if 'output_token_details' not in usage_metadata:
                            usage_metadata['output_token_details'] = {}
                        if 'input_token_details' not in usage_metadata:
                            usage_metadata['input_token_details'] = {}
                        
                        # Update the response object
                        response.usage_metadata = usage_metadata
                
                return {
                    "message": response,
                    "content": response.content,
                    "iteration": iteration_idx,
                    "agent_name": self.name
                }
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "message": None,
                        "content": f"Rate limit error after {max_retries} retries: {str(e)}",
                        "iteration": iteration_idx,
                        "agent_name": self.name,
                        "error": True
                    }
            except Exception as e:
                return {
                    "message": None,
                    "content": f"Error: {str(e)}",
                    "iteration": iteration_idx,
                    "agent_name": self.name,
                    "error": True
                }


class ResultExtractor:
    """Extract and format final results from agent conversations"""
    
    def __init__(self, model_name: str, format_prompt: str = ""):
        self.model_name = model_name
        self.format_prompt = format_prompt
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,
            max_retries=2
        )
        self.name = "result_extractor"

    def extract(self, agent_histories: List[List[Dict]], problem_text: str) -> Dict[str, Any]:
        """Extract final answer from agent conversation histories"""
        # Get the last response from the last agent
        if not agent_histories or not agent_histories[-1]:
            mock_message = type('MockMessage', (), {
                'content': "No solution found",
                'usage_metadata': {
                    'input_tokens': 0, 
                    'output_tokens': 0, 
                    'total_tokens': 0,
                    'input_token_details': {},
                    'output_token_details': {}
                }
            })()
            return {"message": mock_message, "final_answer": "No solution found"}
            
        last_response = agent_histories[-1][-1]
        content = last_response.get("content", "")
        
        # Try to extract and refine answer using format_prompt
        try:
            # Try multiple JSON parsing strategies
            raw_answer = content
            json_content = None
            
            # Helper function to standardize JSON format (same as in _extract_thought_and_answer)
            def standardize_json_extractor(json_str: str) -> str:
                """Standardize JSON string to proper format"""
                # Remove control characters
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Fix single quotes to double quotes (but be careful with content)
                # First protect escaped quotes and content within double quotes
                protected_parts = []
                def protect_quoted_content(match):
                    protected_parts.append(match.group(0))
                    return f"__PROTECTED_{len(protected_parts)-1}__"
                
                # Protect already properly quoted strings
                json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', protect_quoted_content, json_str)
                
                # Now fix single quotes to double quotes for unprotected parts
                json_str = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', json_str)
                
                # Fix unquoted property names
                json_str = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                
                # Restore protected content
                for i, protected in enumerate(protected_parts):
                    json_str = json_str.replace(f"__PROTECTED_{i}__", protected)
                
                # Fix common escape sequence issues carefully
                # Don't double-escape already escaped sequences
                json_str = re.sub(r'(?<!\\)\\n', '\\\\n', json_str)  # Fix literal newlines not already escaped
                json_str = re.sub(r'(?<!\\)\\t', '\\\\t', json_str)  # Fix literal tabs not already escaped
                
                return json_str
            
            # Strategy 1: Enhanced JSON pattern matching
            json_patterns = [
                # More precise patterns for complete JSON objects
                r'\{\s*"[^"]+"\s*:\s*"[^"]*"(?:\s*,\s*"[^"]+"\s*:\s*"[^"]*")*\s*\}',  # Simple key-value pairs
                r'\{(?:[^{}]|"[^"]*")*\}',  # Balanced braces with quoted content
                r'\{.*?\}',  # Simple JSON (non-greedy)
                r'\{.*\}',   # Simple JSON (greedy)
            ]
            
            for pattern in json_patterns:
                try:
                    json_match = re.search(pattern, content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        
                        # Apply standardization
                        json_str = standardize_json_extractor(json_str)
                        
                        json_content = json.loads(json_str)
                        break
                except (json.JSONDecodeError, AttributeError) as e:
                    # Only print if this is not a fallback attempt
                    if "Simple JSON" not in pattern:
                        print(f"JSON parsing failed with pattern {pattern}: {str(e)}")
                    continue
            
            # Extract answer from JSON if successful
            if json_content:
                if "code" in json_content:
                    raw_answer = json_content["code"]
                elif "answer" in json_content:
                    raw_answer = json_content["answer"]
                else:
                    raw_answer = str(json_content)
            
            # Fallback: try simple text extraction
            if not raw_answer or raw_answer == content:
                # Look for code blocks
                code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
                if code_match:
                    raw_answer = code_match.group(1)
                else:
                    raw_answer = content
                
            # Use format_prompt to refine the final answer if available
            if self.format_prompt and raw_answer:
                extraction_prompt = f"""Original problem: {problem_text}

Agent's solution:
{raw_answer}

Please extract and format the final answer according to the following requirements:
{self.format_prompt}

Provide only the final answer in the required format."""

                messages = [
                    SystemMessage(content="You are a professional result formatter. Extract and format the final answer according to the given requirements."),
                    HumanMessage(content=extraction_prompt)
                ]
                
                try:
                    response = self.llm.invoke(messages)
                    final_answer = response.content.strip()
                    
                    # Ensure response has proper usage_metadata format
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        usage_metadata = response.usage_metadata
                        if isinstance(usage_metadata, dict):
                            if 'output_token_details' not in usage_metadata:
                                usage_metadata['output_token_details'] = {}
                            if 'input_token_details' not in usage_metadata:
                                usage_metadata['input_token_details'] = {}
                            response.usage_metadata = usage_metadata
                    
                    return {
                        "message": response,
                        "final_answer": final_answer
                    }
                except Exception as e:
                    print(f"Format extraction failed: {str(e)}, using raw answer")
                    # Fallback to raw answer
                    pass
            
            # Fallback: use raw answer
            final_answer = raw_answer
            
        except Exception as e:
            print(f"Answer extraction failed: {str(e)}")
            final_answer = content
            
        # Create a mock message for compatibility
        mock_message = type('MockMessage', (), {
            'content': final_answer,
            'usage_metadata': {
                'input_tokens': 0, 
                'output_tokens': 0, 
                'total_tokens': 0,
                'input_token_details': {},
                'output_token_details': {}
            }
        })()
        
        return {
            "message": mock_message,
            "final_answer": final_answer
        }


class ADAS(AgentSystem):
    """
    ADAS (Agentic Decision-Making and Self-Correction) Framework
    
    Implements a meta-agent that uses iterative "try-reflect-retry" process
    to systematically explore problem solutions with memory and reflection.
    """
    
    def __init__(self, name: str = "adas", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", 5)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.7)
        self.format_prompt = getattr(self, 'format_prompt', "")
        
        # Initialize components
        agent_components = self._create_agents()
        self.agents = [w for w in agent_components["workers"] if isinstance(w, Agent)]
        extractors = [w for w in agent_components["workers"] if isinstance(w, ResultExtractor)]
        if extractors:
            self.extractor = extractors[0]
        else:
            self.extractor = ResultExtractor(self.model_name, self.format_prompt)

    def _create_agents(self) -> Dict[str, List]:
        """Create agents and result extractor for tool integration support"""
        # Create the meta-agent
        meta_agent = Agent(
            agent_id="meta_agent",
            name="ADAS Meta Agent",
            model_name=self.model_name,
            system_prompt=ADASPrompts.SYSTEM_PROMPT
        )
        
        # Create result extractor
        extractor = ResultExtractor(self.model_name, self.format_prompt)
        
        return {
            "workers": [meta_agent, extractor]
        }

    def _format_memory_for_reflection(self, memory: List[Dict[str, Any]]) -> str:
        """Format memory for reflection prompts"""
        if not memory:
            return "No previous attempts."
            
        formatted_memory = []
        for i, attempt in enumerate(memory):
            formatted_memory.append(f"""
Attempt {i + 1}:
Thought: {attempt.get('thought', 'N/A')}
Proposed Solution: {attempt.get('answer', 'N/A')}
Score: {attempt.get('score', 0)}
Feedback: {attempt.get('feedback', 'N/A')}
---""")
        
        return "\n".join(formatted_memory)

    def _extract_thought_and_answer(self, response_content: str) -> tuple[str, str]:
        """Extract thought process and answer from agent response"""
        # Clean the response content
        content = response_content.strip()
        
        # Simplified JSON extraction logic - try direct parsing first
        json_content = None
        
        # Strategy 1: Direct parsing of entire content
        try:
            json_content = json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"✗ Direct parsing failed: {str(e)}")
            
            # Strategy 2: Find JSON boundaries and extract
            json_start = content.find('{')
            json_end = content.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_candidate = content[json_start:json_end + 1]
                print(f"Extracted JSON candidate: {len(json_candidate)} characters")
                
                try:
                    json_content = json.loads(json_candidate)
                  
                except json.JSONDecodeError as e:
                    print(f"✗ JSON candidate parsing failed: {str(e)}")
                    print(f"First 100 chars of JSON candidate: {repr(json_candidate[:100])}")
        
        # Extract information from JSON
        if json_content and isinstance(json_content, dict):
            thought = json_content.get("thought", "")
            answer = json_content.get("code", json_content.get("answer", ""))
            
            if thought or answer:
               
                return thought, answer
        
        # Fallback strategy: Text pattern extraction
        print("Using text pattern extraction...")
        
        thought = ""
        answer = ""
        
        # Simple text pattern matching
        # Look for "thought": "content"
        thought_match = re.search(r'"thought":\s*"([^"]*(?:\\.[^"]*)*)"', content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            print(f"Text pattern found thought: {len(thought)} chars")
        
        # Look for "code": "content"
        code_match = re.search(r'"code":\s*"([^"]*(?:\\.[^"]*)*)"', content, re.DOTALL)
        if code_match:
            answer = code_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            print(f"Text pattern found code: {len(answer)} chars")
        
        # Final fallback
        if not thought and not answer:
            answer = content
            print("Using entire content as answer")
        
        return thought, answer

    def _check_correctness(self, answer: str, reference: str = None) -> float:
        """Basic correctness check - can be overridden for specific tasks"""
        if not answer or answer.strip() == "":
            return 0.0
        
        # For ADAS, we use a simple heuristic based on response quality
        # In practice, this would be task-specific
        if len(answer.strip()) > 10:  # Has substantial content
            return 0.8
        return 0.3

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the ADAS meta-agent with iterative reflection process
        
        Args:
            problem: Problem dictionary containing the task to solve
            
        Returns:
            Dictionary with messages and final_answer
        """
        problem_text = problem.get("problem", str(problem))
        all_messages = []
        memory = []
        
        # Get the meta-agent
        meta_agent = self.agents[0] if self.agents else None
        if not meta_agent:
            return {
                "messages": [],
                "final_answer": "Error: No meta-agent available"
            }

        best_solution = None
        best_score = -1
        
        for iteration in range(self.max_iterations):
            try:
                # Choose prompt based on iteration
                if iteration == 0:
                    # First attempt - use main prompt
                    prompt = f"{ADASPrompts.MAIN_PROMPT}\n\nProblem: {problem_text}"
                else:
                    # Subsequent attempts - use reflection prompt
                    context = self._format_memory_for_reflection(memory)
                    if iteration == 1:
                        prompt = f"{ADASPrompts.REFLECTION_PROMPT_R1}\n\nPrevious attempts:\n{context}\n\nOriginal problem: {problem_text}"
                    else:
                        prompt = f"{ADASPrompts.REFLECTION_PROMPT_R2}\n\nPrevious attempts:\n{context}\n\nOriginal problem: {problem_text}"

                # Generate response
                response_data = meta_agent.generate_response(prompt, iteration)
                
                if response_data.get("error"):
                    # Handle error with error reflection prompt
                    error_msg = response_data.get("content", "Unknown error")
                    error_prompt = ADASPrompts.ERROR_REFLECTION_PROMPT.format(error_msg=error_msg)
                    response_data = meta_agent.generate_response(error_prompt, iteration)

                # Store message
                if response_data.get("message"):
                    all_messages.append(response_data["message"])

                # Extract thought and answer
                content = response_data.get("content", "")
                thought, answer = self._extract_thought_and_answer(content)
                
                # Evaluate solution
                score = self._check_correctness(answer, problem.get("solution"))
                
                # Update memory
                current_trial = {
                    "iteration": iteration,
                    "prompt": prompt,
                    "thought": thought,
                    "answer": answer,
                    "score": score,
                    "feedback": f"Score: {score:.2f}"
                }
                memory.append(current_trial)
                
                # Update best solution
                if score > best_score:
                    best_score = score
                    best_solution = answer
                
                # Early termination if good enough
                if score >= 0.9:
                    break
                    
            except Exception as e:
                error_msg = f"Error in iteration {iteration}: {str(e)}"
                mock_message = type('MockMessage', (), {
                    'content': error_msg,
                    'usage_metadata': {
                        'input_tokens': 0, 
                        'output_tokens': 0, 
                        'total_tokens': 0,
                        'input_token_details': {},
                        'output_token_details': {}
                    }
                })()
                all_messages.append(mock_message)
                break

        # Extract final answer using extractor
        try:
            # Prepare agent history with the best solution
            agent_histories = [[{"content": best_solution or "No solution found"}]]
            extractor_result = self.extractor.extract(agent_histories, problem_text)
            
            # Add extractor message to all_messages if available
            if extractor_result.get("message"):
                all_messages.append(extractor_result["message"])
            
            final_answer = extractor_result.get("final_answer", best_solution or "No solution found")
            
        except Exception as e:
            print(f"Final answer extraction failed: {str(e)}")
            final_answer = best_solution or "No solution found"
            
            # Create error message for compatibility
            error_message = type('MockMessage', (), {
                'content': final_answer,
                'usage_metadata': {
                    'input_tokens': 0, 
                    'output_tokens': 0, 
                    'total_tokens': 0,
                    'input_token_details': {},
                    'output_token_details': {}
                }
            })()
            all_messages.append(error_message)

        return {
            "messages": all_messages,
            "final_answer": final_answer,
            "memory": memory,
            "best_score": best_score
        }


# Register the ADAS system with the framework
AgentSystemRegistry.register(
    "adas",
    ADAS,
    max_iterations=5,
    temperature=0.7
) 