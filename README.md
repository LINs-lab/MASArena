# Multi-Agent System Benchmark Project

## Project Overview
This project aims to develop a comprehensive benchmarking framework for multi-agent systems with high extensibility. The framework allows for easy integration of various agent systems and benchmarks to evaluate multi-agent performance across four key dimensions:

1. **Task Performance**: Measuring how effectively agents complete assigned tasks
2. **Throughput**: Quantifying the volume of tasks processed within a timeframe
3. **Latency**: Measuring response times and processing delays
4. **Resource Cost**: Analyzing computational resources required for operation

Our framework distinguishes between performance gains (dimensions 1-2) and operational costs (dimensions 3-4) to provide a balanced evaluation.

## Important Resources

1. Dataset selection: https://ocnfww8fyyv6.feishu.cn/sheets/QQ3As1dCyhzXnbtqIWfcZU1EnDh
2. MAS benchmark: https://ocnfww8fyyv6.feishu.cn/sheets/Vtw2snMazhHRWStV3i5c7mYZn8d
3. Task Management: https://ocnfww8fyyv6.feishu.cn/base/Ut81bCzLJa1YovsDNX7cfyybnde?table=tblySUDMSSo30bCA&view=vew2hRFW2F
4. Model Table (Hypothesis): https://ocnfww8fyyv6.feishu.cn/sheets/DtE7s0HAHhJoDftFrINcUbsOnLe

## Current Status
- [x] Core workflow implementation for benchmark integration and token consumption tracking
- [x] Agent system integration
- [x] Performance metrics calculation based on token details
- [x] Visualization of the metrics and agent interaction.
- [ ] MCP tools integration
- [ ] Benchmark suite integration (without MCP tools)
- [ ] Benchmark suite integration (with MCP tools)
- [ ] Runtime monitoring system for throughput, latency, and resource usage in deployment scenarios
- [ ] Clear definition of agent workflow and MAS. 

## Timeline and Milestones

### Phase 1: Framework Development (Completed)
- Core architecture design
- Workflow implementation for benchmark integration
- Token consumption tracking mechanisms
- Complete integration of 2 agent simple systems and 1 benchmark

### Phase 2: Metrics Design (Completed)
- Design the metrics for the benchmark, including throughput, latency, and resource cost.
- Evaluation demo for the metrics.
- Visualization of the metrics and agent interaction.
  
### Phase 3: Integration (April 30 - May 10, 2025)
- **Milestone 1** (May 10, 2025): Complete integration of almost all agent systems and benchmarks
- **Milestone 2** (May 10, 2025): MCP tools integration
- 
### Phase 4: Performance Measurement (May 10 - May 20, 2025)
- **Milestone 3** (May 20, 2025): Develop runtime monitoring system
- **Milestone 4** (May 20, 2025): Plan the experiment settings and run the experiments

### Phase 5: Validation and Refinement (May 20 - May 30, 2025)
- **Milestone 5** (May 30, 2025): Evaluation and analysis of the results



## Getting Started
1. Review the meeting notes to understand project history and decisions
2. Check the current milestone tasks
3. Checkout the branch `base-workflow-implementation`. 
4. Run the workflow to test the basic functionality.



  <h1 align="center">MASArena 🏟️</h1>
  <!-- <p align="center"><i>Multi-Agent Systems Arena</i></p> -->
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+" height="20"/></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT" height="20"/></a>
    <a href="https://lins-lab.github.io/MASArena"><img src="https://img.shields.io/badge/📖%20Docs-MASArena-blue" alt="Documentation" height="20"/></a>
    <a href="https://deepwiki.com/LINs-lab/MASArena"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki" height="20"></a>
  </p>
  
  
  <p align="center">
    <b>Layered Architecture</b> • <b>Stack</b> • <b>Swap</b> • <b>Built for Scale</b>
  </p>
  <img src="docs/images/intro.svg" style="display: block; margin: 0 auto; max-width: 100%;" alt="MASArena Architecture"/>
</div>


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


## 🌟 Core Features

* **🧱 Modular Design**: Swap agents, tools, datasets, prompts, and evaluators with ease.
* **📦 Built-in Benchmarks**: Single/multi-agent datasets for direct comparison.
* **📊 Visual Debugging**: Inspect interactions, accuracy, and tool use.
* **🤖 Automated Workflow Optimization**: Automatically optimize agent workflows using LLM-driven evolutionary algorithms.
* **🔧 Tool Support**:  Manage tool selection via pluggable wrappers.
* **🧩 Easy Extensions**: Add agents via subclassing—no core changes.
* **📂 Paired Datasets & Evaluators**: Add new benchmarks with minimal effort.
* **🔍 Failure Attribution**: Identify failure causes and responsible agents.

## 🎬 Demo

See MASArena in action! This demo showcases the framework's visualization capabilities:

https://github.com/user-attachments/assets/b6e56eef-e00e-46bb-97e0-02d2aca47403

## 🚀 Quick Start

### 1. Setup

We recommend using [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Configure Environment Variables

Create a `.env` file in the project root and set the following:

```bash
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1
```

### 3. Running Benchmarks

```bash
# Run a standard benchmark (e.g., math with supervisor_mas agent)
./run_benchmark.sh math supervisor_mas 10

# Run the AFlow optimizer on the humaneval benchmark
./run_benchmark.sh humaneval single_agent 10 "" "" aflow
```
* Supported benchmarks: 
  * Math: `math`, `aime`
  * Code: `humaneval`, `mbpp`
  * Reasoning: `drop`, `bbh`, `mmlu_pro`, `ifeval`, `hotpotqa`
* Supported agent systems: 
  * Single Agent: `single_agent`
  * Multi-Agent: `supervisor_mas`, `swarm`, `agentverse`, `chateval`, `evoagent`, `jarvis`, `metagpt`

## 📚 Documentation

For comprehensive guides, tutorials, and API references, visit our complete [documentation](https://lins-lab.github.io/MASArena).

## ✅ TODOs

* [x] Add asynchronous support for model calls
* [x] Implement failure detection in MAS workflows
* [ ] Add more benchmarks emphasizing tool usage
* [ ] Improve configuration for MAS and tool integration
* [ ] Integrate multiple tools(e.g., Browser, Video, Audio, Docker) into the current evaluation framework
* [ ] Optimize the framework's tool management architecture to decouple MCP tool invocation from local tool invocation
* [ ] Implement more benchmark evaluations(e.g., webArena, SweBench) that requires tool usage
* [ ] Reimplementation of the Dynamic Architecture Paper Based on the Benchmark Framework

## 🙌 Contributing

We warmly welcome contributions from the community!

**📋 For detailed contribution guidelines, testing procedures, and development setup, please see [CONTRIBUTING.md](docs/quick_start/CONTRIBUTING.md).**

You can contribute in many ways:

* 🧠 **New Agent Systems (MAS):**
  Add novel single- or multi-agent systems to expand the diversity of strategies and coordination models.

* 📊 **New Benchmark Datasets:**
  Bring in domain-specific or task-specific datasets (e.g., reasoning, planning, tool-use, collaboration) to broaden the scope of evaluation.

* 🛠 **New Tools & Toolkits:**
  Extend the framework's tool ecosystem by integrating domain tools (e.g., search, calculators, code editors) and improving tool selection strategies.

* ⚙️ **Improvements & Utilities:**
  Help with performance optimization, failure handling, asynchronous processing, or new visualizations.

### Quick Start for Contributors

1. **Fork and Clone**: Fork the repository and clone it locally
2. **Setup Environment**: Install dependencies with `pip install -r requirements.txt`
3. **Run Tests**: Execute `pytest tests/` to ensure everything works
4. **Make Changes**: Implement your feature with corresponding tests
5. **Submit PR**: Create a pull request with a clear description

Our automated CI/CD pipeline will run tests on every pull request to ensure code quality and reliability.

