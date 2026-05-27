# BenchAgent

A workflow-evaluation framework for comparing LLM agent systems under a shared
protocol. BenchAgent normalizes benchmark loading, tool access, answer contracts,
usage accounting, and trajectory logging so that single-agent, fixed MAS, and
evolving MAS workflows can be compared on process-level signals — not just final
accuracy.

## 🚀 Quick Start

```bash
uv sync
cp .env.example .env
# → set OPENAI_API_KEY and MODEL_NAME

./run_benchmark.sh                          # math + BenchAgent, all defaults
BENCHMARK=gaia AGENT_SYSTEM=EvoAgent ./run_benchmark.sh
BENCHMARK=mmlu_pro LIMIT=50 MODEL_NAME=gpt-4o ./run_benchmark.sh
```

`run_benchmark.sh` wraps `uv run python main.py` — no venv activation needed.
Every option is an env var; see `.env.example` for the full list.

## 📊 Benchmarks

`math` · `aime` · `gsm8k` · `drop` · `bbh` · `mmlu_pro` · `humaneval` · `mbpp`
`hotpotqa` · `ifeval` · `gaia` · `swebench_lite`

Run `uv run python main.py --help` for the live registry.

## 🤖 Workflows

`BenchAgent` · `Jarvis` · `EvoAgent` · `ChatEval` · `AutoGen` · `Camel`
`LLMDebate` · `MetaGPT` · `AgentVerse` · `Swarm` · `SingleAgent`
`SupervisorMAS` · `MAD` · `ChatDev` · `SimpleSmolagents`

## 🔧 How It Works

```
Benchmark data  ─→  BenchmarkRunner  ─→  Workflow (AgentSystem)
  (shared loader)    (async + semaphore)    (single / fixed MAS / evolving)
                          │                        │
                     Evaluator scores          Trajectory logged
                (@register_benchmark)    (messages, tools, tokens)
                          │
                     summary.json + results.json
```

## ➕ Adding a Workflow

Subclass `AgentSystem` in `mas_arena/agents/` — auto-discovered on import.
Accept `model_name`, `api_key`, `api_base` from `self.config`.

## ➕ Adding a Benchmark

Subclass `BaseEvaluator`, decorate with `@register_benchmark(name="…")`,
drop the file in `mas_arena/evaluators/`.

## 📁 Project Structure

```
.
├── .env.example             # 40+ env vars
├── run_benchmark.sh         # Entrypoint
├── main.py                  # Python entrypoint (argparse)
├── pyproject.toml           # Dependencies
│
├── mas_arena/               # Core framework
│   ├── benchmark_runner.py  #   Execution engine
│   ├── agents/              #   Workflow implementations
│   ├── evaluators/          #   Per-benchmark evaluators
│   ├── tools/ tools_old/    #   Tool implementations
│   ├── memory/              #   Memory systems
│   ├── mcp_servers/         #   MCP tool servers
│   ├── agent_flow/          #   Workflow graph & optimization
│   ├── train/               #   RL training
│   └── optimizers/          #   Optimizers
│
├── data/                    # Benchmark datasets
├── scripts/                 # Launch scripts
└── tests/                   # Test suite
```

## 📄 License

MIT
