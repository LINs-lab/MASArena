# Supported

## 📊 Supported Benchmarks

| Benchmark   | Description                  | Dataset File               |
| ----------- | ---------------------------- | -------------------------- |
| `math`      | Mathematical problem solving | `math_test.jsonl`          |
| `humaneval` | Python code generation       | `humaneval_test.jsonl`     |
| `mbpp`      | Python programming problems  | `mbpp_test.jsonl`          |
| `drop`      | Reading comprehension        | `drop_test.jsonl`          |
| `bbh`       | Complex reasoning tasks      | `bbh_test.jsonl`           |
| `ifeval`    | Instruction following        | `ifeval_test.jsonl`        |
| `aime`      | Math competition problems    | `aime_*_test.jsonl`        |
| `mmlu_pro`  | Multi-domain knowledge       | `mmlu_pro_test.jsonl`      |

## 🤖 Supported Agent Systems

| Agent System     | File                | Description                         |
| ---------------- | ------------------- | ----------------------------------- |
| `single_agent`   | `single_agent.py`   | Single LLM agent                    |
| `supervisor_mas` | `supervisor_mas.py` | Supervisor-based multi-agent system |
| `swarm`          | `swarm.py`          | Swarm-based agent system            |
| `agentverse`     | `agentverse.py`     | Dynamic recruitment agent system    |
| `chateval`       | `chateval.py`       | Debate-based multi-agent system     |
| `evoagent`       | `evoagent.py`       | Evolutionary agent system           |
| `jarvis`         | `jarvis.py`         | Task-planning agent system          |
| `metagpt`        | `metagpt.py`        | Code generation agent system        |