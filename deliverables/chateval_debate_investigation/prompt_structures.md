# New Core MAS Prompt 结构详解

## 0. 共享基础设施

### prompts.yaml（BenchAgent 核心模板）

| Key | 用途 |
|-----|------|
| `base_manager_prompt` | Manager CodeAgent 系统提示：工具优先级、文件访问、search_agent 委托、SymPy 超时 |
| `search_agent_prompt_managed_task` | Search Agent 任务提示 + final_answer 格式 |
| `build_search_keywords_prompt` | 记忆检索前的关键词/计划生成 |
| `semantic_match_prompt` | 答案语义匹配校验 |
| `failure_attribution_and_suggestion_prompt` | 失败归因与改进建议 |
| `extract_key_steps` | 执行日志摘要 |

### format_prompts.py

按 benchmark（`bbh`, `gaia`, `math`, `mmlu`, ...）注入 `<answer>...</answer>` 等格式约束，通过 `AgentSystem.format_prompt` 传入各 MAS。

### bench_agent.py 组装逻辑

```text
manager_system_prompt =
  CodeAgent 默认 system_prompt
  + base_manager_prompt
  + additional_instructions   # ← 各 MAS 注入点

run_agent user prompt =
  "Answer this question correctly..." + problem + files + format_prompt
  + [memory] + [semantic retry] + [failure retry]

run_agent_step =
  augmented_question（MAS 构造）
  + format_prompt（末尾追加）
```

---

## 1. ChatEval (`chateval_newcore.py`)

### 角色 System Prompt（→ BenchAgent additional_instructions）

**Math Expert**
```
You are a Mathematics Expert. Analyze the problem mathematically.
Use Python to VERIFY calculations if needed, but be CONCISE.
+ FILE PROCESSING GUIDELINES (excel/image/audio/doc tools)
+ DEEP SEARCH GUIDELINES (web_search)
+ manager_tools: python_interpreter 等
```

**Logic Expert**
```
You are a Logic Expert. Analyze the logical structure.
Be EXTREMELY CONCISE. Do not use tools.
+ ANALYSIS FOCUS (dependencies, assumptions, external info)
```

**Critical Thinking Expert**
```
You are a Critical Thinking Expert. Analyze from multiple angles.
Be EXTREMELY CONCISE. Do not use tools.
+ CRITICAL ANALYSIS (alternatives, pitfalls, completeness)
```

### 每轮 Context Prompt（`_build_context`）

```text
Original Problem: {problem}

Discussion History:
{debate_history 或 "No previous discussion."}

{format_prompt}

[可选] ⚠️ FILE PROCESSING REQUIRED: ...

📋 OUTPUT FORMAT REQUIREMENTS:
- Use final_answer tool ...
- Wrap in <answer></answer> if required ...

It is now Round {N}. You are {agent_name}.
Provide your insights. Be CONCISE.
If Math Expert: verify with tools. Others: pure reasoning only.
```

### 合成 Prompt（`_extract_final_result`）

```text
Original problem: {problem}

Debate History:
{history_text}

Synthesize the debate and provide the final answer.
Requirements:
- Choose most reasonable solution
- Verify file/search steps if needed
- ONLY final answer text
- Match exact format
{format_prompt}

IMPORTANT: no refusal text; rescue pass from candidates if needed
```

---

## 2. Jarvis (`jarvis_newcore.py`)

### additional_instructions

```text
You are Jarvis, a highly capable AI assistant...
1. **Analyze & Plan**: outline steps before tools
2. **Execute**: Python, Search, etc.
3. **Verify & Summarize**: clear final answer

+ [可选] config.system_prompt
+ [可选] format_prompt as Output Requirements
```

单 BenchAgent，`run_agent()` 完整 GAIA 流程（非 step 模式）。

---

## 3. LLM Debate (`llm_debate_newcore.py`)

### BenchAgent additional_instructions
`config.additional_instructions` 或 `format_prompt`

### 辩论 Prompt 序列

| 阶段 | Prompt |
|------|--------|
| 首轮 user | `Can you solve the following problem? {question} Please explain... state your final answer clearly` |
| 后续轮 debate | `These are the recent/updated opinions from other agents: Agent N response: ```{answer}``` ... Use these opinions carefully... updated answer?` |
| 聚合 | `Task:\n{query}\n\nSolution 1:\n{answer1}\n... Given all the above solutions, reason over them carefully... {format_prompt}` |

配置：`agents_num`（默认 2）、`rounds_num`（默认 3）

---

## 4. AutoGen (`autogen_newcore.py`)

| Agent | System Prompt |
|-------|---------------|
| primary | `You are a helpful AI assistant, skilled at generating creative and accurate content.` |
| critic | `Provide constructive feedback... Respond with 'APPROVE'...` |

### Primary 每轮输入
```text
System Prompt for primary: {prompt}
--- Conversation History ---
{USER/PRIMARY/CRITIC}: {content}
...
Now, as the PRIMARY agent, generate or revise content...
```

Critic 走 OpenAI API 直接调用（非 BenchAgent）。

---

## 5. CAMEL (`camel_newcore.py`)

| 角色 | System Prompt |
|------|---------------|
| User Evaluator | `You are the Question USER... include 'TASK_FINISHED' when done` + system_prompt + format_prompt |
| Assistant Solver | `You are the ASSISTANT SOLVER...` + system_prompt + format_prompt |
| 总结 BenchAgent | `You are a professional result analyzer...` + 最近 4 条消息 + Requirements |

max_rounds = 3

---

## 6. Swarm (`swarm_newcore.py`)

### SwarmAgent `_create_prompt`
```text
Please solve the following problem:
{problem}
Think carefully step by step...
Agent ID: {agent_id}
```

### Aggregator `aggregate`
```text
I need you to analyze multiple solutions...
The original problem: {problem}
The solutions from different agents: ...
{format_prompt}
```

system_prompt 默认：`You are an intelligent AI assistant specialized in solving problems carefully and step by step.`

---

## 7. EvoAgent (`evoagent_newcore.py`)

### 初始 system_prompt 模板（3 个）
1. mathematics expert
2. logical reasoning expert
3. problem-solving expert

### 求解 prompt（BenchEnhancedAgent）
```text
{system_prompt}

Problem to solve:
{problem}
```

### 进化 meta-prompts（LangChain ChatOpenAI）
- **Crossover**: 合并两父 agent JSON `{name, system_prompt}`
- **Mutation**: 变异父 agent 配置
- **Summary**: 汇总 top-5 答案 + format_prompt

---

## 8. MetaGPT (`metagpt_newcore.py`)

| 角色 | 职责 |
|------|------|
| Product Manager | 需求提取 → `<plan>` + `<answer>` |
| Architect | 实现策略 + 伪代码 |
| Project Manager | 任务分配 |
| Engineer | **BenchAgent** 代码实现 |
| QA Engineer | 测试验证 + 最终代码 |

### Engineer BenchAgent prompt
```text
System Role: {engineer.system_prompt}
Task Context:
Product Requirements: ...
Architecture: ...
Assignment: ...
```

各角色通过 LangChain SystemMessage + HumanMessage 流水线调用。

---

## 9. agent_responses 序列化说明（base.py）

`_record_agent_responses(problem_id, messages)` 遍历 messages：
- LangChain AIMessage → `agent_id`, `content`, `message_type: ai_response`
- dict 消息 → 保留 `role`, `content`, `name`, `message_type`

**ChatEval 缺口**：`debate_history` 在 `run_output` 中返回但未传入 `_record_agent_responses`；修复需在 `evaluate()` 中额外序列化 `debate_history` 字段。
