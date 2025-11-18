# Project Roadmap

This document outlines the planned development timeline and future direction for the Multi-Agent System Arena (MAS-Arena). Our goal is to evolve this framework into a comprehensive and community-driven platform for multi-agent research and evaluation.

---

## v0.1.0 

**Focus:** Initial public release with core framework for evaluating and comparing single and multi-agent systems.

| Category                  | Key Initiatives                                                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Framework**        | - **Modular Design:** Key components like agents, tools, datasets, prompts, and evaluators are designed to be swappable. <br>- Straightforward to add new benchmarks with paired datasets and evaluators. |
| **Benchmarks**            | - **Built-in Benchmarks:** Integrated several benchmarks for direct comparison of agent systems. <br>- **Supported Benchmarks:** `math`, `aime`, `humaneval`, `mbpp`, `drop`, `bbh`, `mmlu_pro`, `ifeval`. |
| **Agent Systems**         | - **Supported Agent Systems:** Includes a single agent baseline and various multi-agent systems: `single_agent`, `supervisor_mas`, `swarm`, `agentverse`, `chateval`, `evoagent`, `jarvis`, `metagpt`. |
| **Tooling**               | - **Pluggable Tool Support:** A wrapper-based system to manage tool selection and integration for agents. |
| **Visualization**         | - **Visual Debugging**: Support for inspecting agent interactions, accuracy, and tool usage. |


---

## v0.2.0

**Focus:** Strengthen the existing foundation, improve developer experience, and expand core features based on initial feedback.

| Category                  | Key Initiatives                                                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Framework**        | - **Configuration Overhaul:** Introduce configuration (e.g., using YAML/Hydra) to simplify managing complex agent and benchmark settings. <br> - **Failure Analysis:** Add a failure analysis plugin to help identify why MAS fails. |
| **Benchmarks**            | - **Enhance with tools:** Integrate multiple tools(e.g., Browser, Video, Audio, Docker) and benchmarks for tool usage, like `swebench`. |
| **Agent Systems**         | - Continuously Integrate New Agent Systems.<br/>- **Standardize Agent Outputs:** Enforce a more standardized output schema for `run_agent` to simplify evaluation logic. |
| **Tooling**               | - **Optimize tool management architecture:** Decouple MCP tool invocation from local tool invocation. <br/>- **Tool Caching:** Implement a caching layer for tool outputs. |
| **Documentation & Community** | -**Tutorials:** Write step by step tutorials on "Adding a New Agent" and "Creating a Custom Benchmark". |

---

## v0.3.0 

**Focus:** Add more complex benchmarks, integrate a wider variety of agent systems, and grow the tool ecosystem.

| Category                  | Key Initiatives                                                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Framework**        | - **Agent-to-Agent Communication:** Develop a standardized internal message-passing. <br/>- **Sandbox Environment:** Enhance the execution environment with finer-grained control. |
| **Tooling**               | - **Tool Discovery & Creation:** Develop an experimental feature where agents can attempt to generate new tools (e.g., Python functions) on the fly and add them to their context.<br/>- **Async Tools:** Refactor the `ToolManager` to fully support asynchronous tool execution. |

---