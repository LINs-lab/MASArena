# 🚀 Extending MASArena Framework

A comprehensive guide to extending MASArena with custom Multi-Agent Systems and Evaluators.

## 📋 Table of Contents

- [🤖 Multi-Agent System Extension](#-multi-agent-system-extension)
  - [📋 Implementation Requirements](#-implementation-requirements)
  - [📝 Implementation Steps](#-implementation-steps)
  - [⚡ Advanced Features](#-advanced-features)
  - [💡 Complete Example](#-complete-example)
- [🎯 Evaluator Extension](#-evaluator-extension)
  - [🔧 Basic Implementation](#-basic-implementation)
  - [⚡ Advanced Features](#-advanced-features-1)
  - [💻 Code Evaluation](#-code-evaluation)
  - [📖 Complete Examples](#-complete-examples)
- [✅ Best Practices](#-best-practices)
- [🚨 Common Issues](#-common-issues)

---

## 🤖 Multi-Agent System Extension

### 📋 Implementation Requirements

**✅ Essential Requirements:**

- Extend `AgentSystem` base class
- Implement `run_agent()` method (abstract method - required)
- Include `evaluator` in config during initialization
- Return proper message format with usage metadata
- Register with `AgentSystemRegistry`

**💡 Optional but Recommended:**

- Implement `_create_agents()` for tool integration support
- Use `self.format_prompt` for benchmark-specific formatting
- Handle async execution properly if needed

### 📝 Implementation Steps

#### Step 1: Create Agent System Class Structure

✅ Langgraph supported
✅ Customizable agent and multi-agent interaction

**📋 Implementation Guide:**

- Inherit from `AgentSystem` base class
- Initialize configuration parameters (num_agents, num_rounds, model_name)
- Set up agent components using `_create_agents()` method
- Extract workers and result extractors from created components
- Validate that required components are available

**💡 SupervisorMAS Implementation Example (LangGraph Structure):**

```python
# mas_arena/agents/supervisor_mas.py

    def _init_graph_if_needed(self, problem_input: Optional[Any] = None, feedback: Optional[Any] = None):
        if self.graph is not None:
            return

        # _create_agents now returns a dict {"researcher": researcher_node, "coder": coder_node}
        # If wrapped by ToolIntegrationWrapper, the nodes will have been modified in-place.
        worker_nodes_map = self._create_agents(problem_input=problem_input, feedback=feedback)

        research_node_obj = worker_nodes_map.get("researcher")
        coder_node_obj = worker_nodes_map.get("coder")
        
        if not research_node_obj or not coder_node_obj:
            raise RuntimeError("Could not find researcher or coder agent nodes from _create_agents dictionary.")

        builder = StateGraph(State)
        checkpointer = InMemorySaver()

        supervisor_model = self.config.get("supervisor_model_name", self.config.get("model_name", os.getenv("MODEL_NAME", "gpt-4o-mini")))
        builder.add_node("supervisor", create_supervisor(model_name=supervisor_model))
        
        builder.add_node("researcher", research_node_obj)
        builder.add_node("coder", coder_node_obj)

        builder.add_edge(START, "supervisor")
        
        builder.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"researcher": "researcher", "coder": "coder", END: END},
        )
        
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("coder", "supervisor")

        self.graph = builder.compile(checkpointer=checkpointer)

```

**💡 ChatEval Implementation Example (Basic Structure):**

```python
# mas_arena/agents/chateval.py
class ChatEval(AgentSystem):
    """Multi-agent evaluation system based on iterative debate"""
    
    def __init__(self, name: str = "chateval", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.num_rounds = self.config.get("num_rounds", 2)
        self.model_name = get_model_name(self.config.get("model_name"))

        agent_components = self._create_agents()
        self.agents = [w for w in agent_components["workers"] if isinstance(w, Agent)]
        extractors = [
            w for w in agent_components["workers"] if isinstance(w, ResultExtractor)
        ]
        if not extractors:
            raise ValueError(
                "ResultExtractor not found in components created by _create_agents."
            )
        self.extractor = extractors[0]
```

#### Step 2: Implement Core `run_agent` Method

**📋 Implementation Guide:**

- Extract problem text from input dictionary
- Initialize message storage for tracking LLM responses
- Implement multi-round agent interaction logic
- Collect and process agent responses with proper metadata
- Extract final answer using result extractor
- Return formatted result with messages and final answer

**💡 ChatEval Implementation Example (run_agent Core Method):**

```python
# mas_arena/agents/chateval.py
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run iterative debate process"""
        problem_text = problem["problem"]

        # store all LLM response objects
        all_messages = []
        # Store all responses for context building - indexed by [round][agent_index]
        round_responses = []

        # iterative discussion process
        for t in range(self.num_rounds):
            # Clear all agents' chat history at the start of each round to prevent accumulation
            for agent in self.agents:
                agent.chat_history = []
            
            # Store responses for this round
            current_round_responses = []
            
            for n, agent in enumerate(self.agents):
                # Rebuild context from scratch using all previous responses
                context = self._build_context(problem_text, n, t, round_responses)
                response_data = await agent.generate_response(context)

                # save response object
                if "message" in response_data:
                    all_messages.append(response_data["message"])

                # Store this agent's response for current round
                solution_text = response_data.get("solution", "")
                current_round_responses.append({
                    "agent_index": n,
                    "agent_name": AGENT_NAMES[n],
                    "response": solution_text
                })
            
            # Add current round responses to the overall history
            round_responses.append(current_round_responses)

        # Generate conversation history from round responses
        conversation_history = self._generate_conversation_history(round_responses)
        
        # extract final answer using ResultExtractor
        extractor_result = await self.extractor.extract(conversation_history, problem_text)

        # add evaluator message
        if "message" in extractor_result and extractor_result["message"]:
            all_messages.append(extractor_result["message"])
        
        return {
            "messages": all_messages,
            "conversation_history": conversation_history,
            "final_answer": extractor_result["message"].content if extractor_result["message"] else None,
        }
```

#### Step 3: Implement `_create_agents` Method (Tool Integration Support)

**📋 Implementation Guide:**

- Create specialized `AgentNode` instances for each role
- Set agent names, models, and system prompts
- Create result extractor with format prompt integration
- Return dictionary with "workers" key containing all components
- Ensure each worker has `.name` and `.llm` attributes for tool binding

**💡 ChatEval Implementation Example (_create_agents Tool Integration):**

```python
# mas_arena/agents/chateval.py
    def _create_agents(self) -> Dict[str, List[Any]]:
        """Create multiple agent instances and result extractor"""
        debate_agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                name=AGENT_NAMES[i],
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i),
            )
            debate_agents.append(agent)

        # Create and assign the extractor here  
        extractor = ResultExtractor(self.model_name, getattr(self, 'format_prompt', ''))

        return {"workers": debate_agents + [extractor]}
```

#### Step 4: Register System with Framework

**📋 Implementation Guide:**

- Use `AgentSystemRegistry.register()` to make system available
- Provide system name as string identifier
- Pass class reference (not instance)
- Include default configuration parameters
- These defaults can be overridden during initialization

**💡 ChatEval Implementation Example (Registration):**

```python
# mas_arena/agents/chateval.py
# register agent system
AgentSystemRegistry.register("chateval", ChatEval, num_agents=3, num_rounds=2)
```

### ⚡ Advanced Features

#### 🎨 Format Prompt Integration

**📋 Implementation Guide:**

- Accept `format_prompt` parameter in initialization
- Store format prompt for benchmark-specific requirements
- Use format prompt in result extraction and agent prompts
- Configure timeout and retry settings for robust operation

**💡 ChatEval Implementation Example (Format Prompt Integration):**

```python
# mas_arena/agents/chateval.py
def get_model_name(config_model_name: Optional[str] = None) -> str:
    """Get model name from config or environment variable"""
    return config_model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

def create_llm(model_name: str) -> ChatOpenAI:
    """Create ChatOpenAI instance with standardized configuration"""
    return ChatOpenAI(
        model=model_name,
        timeout=60,  # Set request timeout to 60 seconds
        max_retries=2,  # Set maximum retry attempts to 2
    )

class ResultExtractor:
    def __init__(self, model_name: Optional[str] = None, format_prompt: str = ""):
        self.model_name = get_model_name(model_name)
        self.format_prompt = format_prompt
        self.llm = create_llm(self.model_name)
        self.name = "result_extractor"
```

#### 🤖 Agent Node Pattern

**📋 Implementation Guide:**

- Use dataclass decorator for clean agent definition
- Include required attributes: agent_id, name, model_name, system_prompt
- Initialize chat history as empty list
- Set up LLM instance with timeout and retry configuration
- Ensure compatibility with tool integration framework

**💡 ChatEval Implementation Example (Agent Class Definition):**

```python
# mas_arena/agents/chateval.py
from dataclasses import dataclass

@dataclass
class Agent:
    """Represents an LLM agent"""

    agent_id: int
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []
        self.llm = create_llm(self.model_name)
```

#### 🔄 Usage Metadata Handling

**📋 Implementation Guide:**

- For native OpenAI API calls or non-structured output: No manual handling required
- For structured output: Use `self.llm.with_structured_output(schema=AgentResponse, include_raw=True)`
- Usage metadata is automatically handled by the framework
- Focus on implementing core agent logic and response processing
- Return standardized response format with message and solution

### 📋 Key Implementation Summary

**🔧 Implementation Points:**

- Inherit from `AgentSystem` base class
- Implement required `run_agent()` method  
- Ensure config includes `evaluator` key
- Return dictionary containing `messages` and `final_answer`
- Optional: Implement `_create_agents()` for tool integration support

**📝 Registration Process:**
Use `AgentSystemRegistry.register()` to register system and provide default configuration parameters.

> 📄 **Complete Implementation Reference**: [`mas_arena/agents/chateval.py`](../mas_arena/agents/chateval.py)

---

## 🎯 Evaluator Extension

### 🔧 Basic Implementation

#### Step 1: Basic Structure and Registration

**📋 Implementation Guide:**

- Use `@register_benchmark` decorator to register evaluator
- Define normalization keys mapping for data field standardization
- Inherit from `BaseEvaluator` base class
- Provide comprehensive docstring explaining evaluator purpose
- Set up evaluator name and supported answer formats

**💡 MMLU_pro Implementation Example (Registration and Class Definition):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
@register_benchmark(
    name="mmlu_pro",
    normalization_keys={
        "id": "id",
        "problem": "question",
        "solution": "answer",
    }
)
class MMLU_ProEvaluator(BaseEvaluator):
    """
    Evaluator for the MMLU Professional mas_arena.
    
    This evaluator assesses agent performance on the MMLU_pro dataset
    using exact matching of answers (A, B, C, etc.).
    """
```

#### Step 2: Initialize Configuration

**📋 Implementation Guide:**

- Call parent class initialization with name and config
- Set up evaluation-specific weights and parameters
- Configure dataset loading and validation
- Set up logging and error handling
- Define evaluation metrics and scoring methods

**💡 MMLU_pro Implementation Example (Initialization):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
    def __init__(self, name="mmlu_pro", config=None):
        """
        Initialize the MMLU Professional evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration dictionary containing:
                - data_path: Path to the MMLU_pro dataset
                - log_path: Path to save evaluation logs
        """
        super().__init__(name, config or {})
        
        # Weight for exact match score is always 1.0 as it's the only metric
        self.exact_match_weight = 1.0
        
        # Load the dataset
        self._load_dataset()
```

#### Step 3: Implement Core Evaluation Method

**📋 Implementation Guide:**

- Extract final answer and reference solution from inputs
- Use specialized answer extraction method for response parsing
- Apply scoring logic (exact match, numerical comparison, etc.)
- Calculate evaluation metrics and scores
- Return standardized evaluation results dictionary
- Include extracted answer and original final answer

**💡 MMLU_pro Implementation Example (evaluate Method):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an agent's solution to a MMLU_pro problem.
        
        Args:
            problem: Problem dictionary containing:
                - question: Problem text (with options)
                - answer: Correct answer (letter)
                - answer_index: Index of correct answer (optional)
            run_result: Results from agent's execution, containing:
                - final_answer: Agent's final answer text
                - messages: Agent's message history
            
        Returns:
            Evaluation results
        """
        final_answer = run_result.get("final_answer", "")
        reference_letter = problem.get("solution", "")
        
        # Extract the final letter from the agent's response
        extracted_answer = self.extract_answer_from_response(final_answer)
        
        # Calculate exact match score (letter-based)
        score = self.check_exact_match(reference_letter, extracted_answer)
        
        # Record evaluation results
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        }
```

### ⚡ Advanced Features

#### 🔍 Answer Extraction

**📋 Implementation Guide:**

- Use regular expressions to extract formatted answers
- Handle multiple answer formats (tags, patterns, raw text)
- Implement fallback strategies for unformatted responses
- Clean and normalize extracted text
- Support flexible answer parsing for different benchmarks

**💡 MMLU_pro Implementation Example (Answer Extraction):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer from agent response.
        
        Args:
            response: Complete response text from agent
            
        Returns:
            Extracted answer letter
        """
        # Try to extract answer from <answer> tags, allowing for whitespace
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no tags found, return original response
        return response.strip()
```

#### ✅ Answer Verification

**📋 Implementation Guide:**

- Implement case-insensitive comparison for text answers
- Handle numerical index to letter conversion (1→A, 2→B, etc.)
- Apply normalization and cleaning to both reference and candidate
- Return numerical score (1.0 for match, 0.0 for no match)
- Include error handling for malformed inputs

**💡 MMLU_pro Implementation Example (Exact Match Verification):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
    def check_exact_match(self, reference: str, candidate: str) -> float:
        """
        Check if the candidate exactly matches the reference (case-insensitive).
        
        Args:
            reference: Reference answer (e.g., 'A', 'B', 'C', etc.)
            candidate: Candidate answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Clean and normalize both answers
        ref_clean = reference.strip().upper()
        cand_clean = candidate.strip().upper()
        
        # Check for exact match
        if cand_clean == ref_clean:
            return 1.0
        
        # Check if candidate is an index (e.g., "1", "2", "3") converted to letter
        try:
            if cand_clean.isdigit():
                cand_index = int(cand_clean) - 1
                cand_letter = chr(ord('A') + cand_index)
                if cand_letter == ref_clean:
                    return 1.0
        except Exception:
            pass
            
        return 0.0
```

#### 📊 Batch Evaluation

**📋 Implementation Guide:**

- Iterate through all problems in the batch
- Extract problem IDs and reference answers for each item
- Apply evaluation logic consistently across all problems
- Collect comprehensive results with metadata
- Log evaluation progress and summary statistics
- Return standardized results format for benchmark runner

**💡 MMLU_pro Implementation Example (Batch Evaluation):**

```python
# mas_arena/evaluators/mmlu_pro_evaluator.py
    def batch_evaluate(self, problems: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of problems.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Evaluate each problem individually
        for i, problem in enumerate(problems):
            problem_id = problem.get("id", problem.get("question_id", f"unknown_{i}"))
            reference_letter = problem.get("solution", problem.get("answer", ""))
            reference_text = self.get_correct_answer_text(problem)
            response = problem.get("response", "")
            
            # Calculate exact match score
            exact_match = self.check_exact_match(reference_letter, response)
            
            # Record results
            result = {
                "problem_id": problem_id,
                "exact_match": exact_match,
                "combined_score": exact_match,  # Combined score is just the exact match
                "extracted_answer": response,
                "reference_answer": reference_letter,
                "reference_text": reference_text,
                "execution_time_ms": 0,  # Will be updated by the benchmark runner
                "math_score": 1.0 if exact_match >= 0.9 else 0.0  # For compatibility with benchmark runner
            }
            
            results.append(result)
            
            # Log the results
            self.logger.info(f"Problem {problem_id}: Exact={exact_match:.1f}, Combined={exact_match:.4f}")
        
        return results
```

### 💻 Code Evaluation

**🔧 Code Evaluator Key Points:**

- Inherit from `BaseCodeEvaluator` base class (not BaseEvaluator)
- Implement `check_solution(code, test, entry_point)` method
- Implement `extract_code(text)` to extract code from responses
- Must include timeout protection mechanisms
- Use isolated environments for code execution

**📊 Core Process Flow:**

1. **Code Extraction** - Extract Python code from agent responses
2. **Environment Isolation** - Create secure execution environment
3. **Test Execution** - Run test cases to verify code correctness
4. **Timeout Control** - Prevent infinite loops or long execution

### 📋 Evaluator Implementation Summary

**🔧 Core Components:**

- Use `@register_benchmark` decorator for registration
- Inherit from `BaseEvaluator` base class
- Implement required `evaluate()` method
- Configure `normalization_keys` for data mapping
- Optional: Implement answer extraction and verification methods

**📊 Evaluation Process:**

1. **Data Normalization** - Map fields using normalization_keys
2. **Answer Extraction** - Extract final answer from messages
3. **Answer Verification** - Compare predicted vs reference answers
4. **Result Return** - Return score, extracted_answer, final_answer fields

> 📄 **Complete Implementation References**:
>
> - Text Evaluator: [`mas_arena/evaluators/mmlu_pro_evaluator.py`](../mas_arena/evaluators/mmlu_pro_evaluator.py)
> - Code Evaluator: [`mas_arena/evaluators/humaneval_evaluator.py`](../mas_arena/evaluators/humaneval_evaluator.py)

---

## ✅ Best Practices

### 🚀 Performance & Security

- **⚡ Batch Processing**: Implement `batch_evaluate()` for better performance
- **⏱️ Timeout Handling**: Always set timeouts for external calls and code execution
- **🔍 Input Validation**: Validate all inputs before processing
- **🛡️ Error Handling**: Implement comprehensive exception handling
- **📝 Logging**: Add detailed logging for debugging and monitoring

### 🧪 Testing & Validation

- **🎯 Unit Tests**: Test individual components thoroughly
- **🔄 Integration Tests**: Test full evaluation pipeline
- **⚠️ Edge Cases**: Test with malformed inputs and edge cases
- **📊 Performance Tests**: Benchmark evaluation speed for large datasets

---

## 🚨 Common Issues

### 📋 Implementation Checklist

**For MAS Extensions:**

- [ ] ✅ Config includes `evaluator` key
- [ ] 📊 Messages have `usage_metadata` for token tracking
- [ ] 🏷️ Agents have `name` and `llm` attributes (for tool integration)
- [ ] ⚡ `run_agent` should be async
- [ ] 📤 Return format includes `messages` and `final_answer`
- [ ] 📋 Proper registration with `AgentSystemRegistry`

**For Evaluator Extensions:**

- [ ] 🎯 Used `@register_benchmark` decorator
- [ ] ✅ Implemented `evaluate` method
- [ ] 🗝️ Proper normalization_keys mapping
- [ ] 🛡️ Error handling for malformed inputs
- [ ] ⏱️ Timeout handling for long operations
