# Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows

Yuhang Fu, Ruishan Fang, Jiaqi Shao, Huiyu Zheng, Zhengtao Zhu, Bing Luo, Tao Lin

> Does adding more agents help an LLM workflow once compared systems share the same
> benchmark loader, tool access, answer contract, usage accounting, and trajectory logging?
> We introduce **BenchAgent**, an evaluation framework that normalizes single-agent,
> fixed MAS, and evolving MAS workflows under one execution protocol.
>
> Under substrate-internal conditions with GPT-4.1, only 1 of 6 tested MAS exceeds the
> matched single-agent anchor on benchmark-balanced average accuracy (EvoAgent, +1.44 pts,
> within Wilson guidance), while the remaining five trail by 2.56--11.29 points.
> On the PAE GAIA snapshot, a Claude-Code-style runtime workflow reaches 66.72% overall
> and 69.23% on Level 3 — 20+ points above the strongest non-Claude baseline.

## 📄 Paper

- `submissions/arxiv/` — arXiv-ready source
- `submissions/emnlp/` — EMNLP-formatted source

## 🚀 Quick Start

BenchAgent and the paper implementation are maintained on the
[`BenchAgent`](https://github.com/LINs-lab/project_multi_agents_benchmark/tree/BenchAgent)
branch.

```bash
git checkout BenchAgent
uv sync
cp .env.example .env
```

Set your API key in `.env`:

```bash
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
```

Run the default benchmark:

```bash
./run_benchmark.sh
```

By default, this runs the `math` benchmark with `bench_agent`, async execution,
and the model specified by `MODEL_NAME`. Common overrides are passed as
environment variables:

```bash
BENCHMARK=gaia LIMIT=20 ./run_benchmark.sh
BENCHMARK=mmlu_pro LIMIT=50 MODEL_NAME=gpt-4o ./run_benchmark.sh
BENCHMARK=gaia AGENT_SYSTEM=evoagent CONCURRENCY=8 ./run_benchmark.sh
MANAGER_TOOLS=python_interpreter SEARCH_TOOLS=search,browser ./run_benchmark.sh
```

Results are written under `results/`, and logs are written under `logs/`.

## 🧭 What Is BenchAgent?

BenchAgent is the evaluation substrate used in the paper. Its goal is not to
make one agent look better, but to make different workflows comparable under the
same protocol: benchmark loading, tool access, answer formatting, token
accounting, timeout handling, and trajectory logging.

The framework supports single-agent baselines, fixed multi-agent systems, and
evolving multi-agent systems through the same `AgentSystem` interface. This lets
the paper compare workflows by both final accuracy and process-level signals,
instead of relying on each system's custom runner.

## 🧩 Implementation Overview

The benchmark entrypoint is `run_benchmark.sh`, which wraps:

```bash
uv run python main.py
```

`main.py` parses benchmark, workflow, model, tool, concurrency, and timeout
options, then creates a `BenchmarkRunner`. The runner loads the selected
benchmark evaluator and agent system from registries, normalizes every problem
record, executes the workflow, scores the output, and writes a structured result.

```text
data/*.jsonl
   -> main.py
   -> BenchmarkRunner
   -> AgentSystem.run_agent()
   -> evaluator.evaluate()
   -> results/*.json + *_summary.json + trajectory artifacts
```

For the BenchAgent workflow itself, the implementation lives in
`mas_arena/agents/bench_agent.py`. It builds a two-level agent structure:

- a manager `CodeAgent` that plans, reasons, writes executable snippets when
  useful, and decides when to call tools or delegate;
- a delegated `search_agent` implemented as a `ToolCallingAgent` for web search,
  browsing, Wikipedia lookup, and document-style information gathering;
- a shared OpenAI-compatible model wrapper with retry handling and configurable
  `api_base`, model name, and timeout settings;
- configurable tool lists for manager and search agents, expanded from names
  such as `python_interpreter`, `search`, `browser`, `wikipedia`, and
  `final_answer`;
- optional memory integration for storing successful and failed trajectories in
  later experimental settings.

Every BenchAgent run returns through the same final-answer contract. The base
`AgentSystem` then records messages, token usage, execution time, trajectory
steps, evaluator score, and error state in a uniform `AgentResult`.

## 📊 Benchmarks And Workflows

The runner discovers benchmarks from `mas_arena/evaluators/` and workflows from
`mas_arena/agents/`. Current benchmark examples include:

`math`, `aime`, `gsm8k`, `drop`, `bbh`, `mmlu_pro`, `humaneval`, `mbpp`,
`hotpotqa`, `ifeval`, `gaia`, and `swebench_lite`.

Workflow examples include:

`bench_agent`, `single_agent`, `jarvis`, `evoagent`, `chateval`, `autogen`,
`camel`, `llm_debate`, `metagpt`, `agentverse`, `swarm`, `supervisor_mas`,
`mad`, and `chatdev`.

Run the live registry with:

```bash
uv run python main.py --help
```

## 📁 Outputs

Each run produces a timestamped result file containing a benchmark summary and
per-problem records. The summary tracks accuracy, total and average duration,
token usage, error counts, and the output path. Per-problem records include the
problem id, expected answer, prediction, score, correctness flag, runtime,
recorded LLM usage, and any saved response or visualization artifact.

When verbose mode is enabled, the runner also prints a visualization command for
the generated summary file.

## 📅 Meeting Notes

| No. | Date | Notes | Feishu summary |
|:--- |:---:|:---:|:---:|
1 | 2025.02.26 20:00 GMT+8 | [Notes](meeting-notes/Meeting-20250226.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/PVXfdIcvYof6R8xKon4cCyQJncb) |
2 | 2025.03.12 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250312.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/AYB8dBd9JoiHK5xRwaacX7QrnLh?from=from_copylink) |
3 | 2025.03.19 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250319.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/GQEYdxeYyo1x1vxukHhcgnlQnMg) |
4 | 2025.03.26 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250326.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/GQEYdxeYyo1x1vxukHhcgnlQnMg) |
5 | 2025.04.02 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250402.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/RCHQd7tuxoDGHtxYXLBcC36Bnsh) |
6 | 2025.04.16 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250416.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/minutes/obcnwx8h58445o46hw12k4w2) |
7 | 2025.04.30 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250430.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/docx/TCUkdxhJGoxPenxKsm0cmKohnjh#doxcnGl4WhCU9SkojzO91IAcltc) |
8 | 2025.05.14 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250514.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/minutes/obcng56h82y11ebj672wg1b2) |
9 | 2025.05.28 20:30 GMT+8 | [Notes](meeting-notes/Meeting-20250528.md) | [Summary](https://ocnfww8fyyv6.feishu.cn/minutes/obcnqgdzxm136p2ja5u53794) |
Feishu Page: [project_multi_agents_benchmark](https://ocnfww8fyyv6.feishu.cn/docx/PVXfdIcvYof6R8xKon4cCyQJncb)
