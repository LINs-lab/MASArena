# Research Focus: Comparative Analysis about when MAS Outperforms Single Agents

* `Key Method`: Evaluate the system **cost** and **performance** trade-off between single agent (SA) and MAS
* `Next Step`: Evaluate the **robustness** and **scalability** of SA and MAS with carefully designed tasks.
* `Optional`: Discuss reasoning model compared with simple model with chain-of-thought or other thinking model


# Related Work


## 1. Single Agent

- Simplicity in design and implementation
- Lower computational requirements
- Ease of control and decision making
- Use cases where single agents excel

[1] [Agent leaderboard](https://huggingface.co/datasets/galileo-ai/agent-leaderboard)

### Related Work (About Evaluation & Benchmarking)


## 2. Multi-Agent Systems (MAS)

- Distribution of tasks and computational load
- Robustness and fault tolerance
- Diverse perspectives and knowledge sharing
- Parallel processing capabilities

### Table of MASs used by [1]

| MAS | Agentic Architecture | Purpose of the System |
| --- | --- | --- |
| MetaGPT  | Assembly Line | Simulating the SOPs of different roles in Software Companies to create open-ended software applications |
| ChatDev | Hierarchical Workflow | Simulating different Software Engineering phases like (design, code, QA) through simulated roles in a software engineering company |
| HyperAgent | Hierarchical Workflow | Simulating a software engineering team with a central Planner agent coordinating with specialized child agents (Navigator, Editor, and Executor) |
| AppWorld | Star Topology | Tool-calling agents specialized to utility services (ex: GMail, Spotify, etc.) being orchestrated by a supervisor to achieve cross-service tasks |
| AG2 | N/A - Agentic Framework | An open-source programming framework for building agents and managing their interactions. |

| MAS | Features | Flexible Agents | Workflow-based |  Limitaiton |
| --- | --- | --- | --- | --- | 
| [Mixture of Agents (ICLR 2025 Spotlight)](https://openreview.net/forum?id=h0ZfDIrj7T) | MOE-like | Yes | Yes | high latency & low throughput |
| [ACC-Collab (ICLR 2025)](https://openreview.net/forum?id=nfKfAzkiez) | DPO | No | No | rely on training samples |
| [EvoMAC (ICLR 2025)](https://arxiv.org/html/2410.16946v1) | textual backpropagation | Yes | Yes (adaptive MAC) | relies on objective environment feedback & code task |
| [ADAS (ICLR 2025)](https://openreview.net/forum?id=t9U3LW7JVX) |Meta Agent Search  | Yes | No | ---|
| [AgentVerse [ICLR 2024]](https://openreview.net/forum?id=EHg5GDnyq1) | role-assign and evaluate | Yes | Yes | --- |
| [Camel (NeurIPS 2023)](https://openreview.net/forum?id=B1l83RkFvH) | role play framework | Yes | Yes | hard to config more agents |
<!-- | [IoA (ICLR 2025 Spotlight)](https://arxiv.org/abs/2407.07061) | System framework | -- | -- |  -- | -->





[1] [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)

## 3. Evaluation & Benchmarking



### Summary of benchmarks and avaliable evaluation framework

| Dataset | Task | ... |
| --- | --- | --- |




### Our proposed evaluation metrics
a. [Throughput](https://www.vellum.ai/llm-leaderboard)
b. Memory and Computational Cost by Model Size [openllm leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)
c. Latency e.g. [TTFT](https://www.vellum.ai/llm-leaderboard)
d. [CO2 emission](benchmark/data/leaderboard/llm_cost_CO2.json)


e. **Robustness**: A system's ability to maintain performance despite errors, unexpected inputs, or perturbations
 * Test distributed resilience by simulating agent failures
 * Complexity of the task

f. **Scalability**: The system's ability to handle increased loads efficiently
 * SA: Perfmance with increasing number of resources (computation, memory, etc.)
 * MAS: Performance with increasing number of agents



**Standard performance metrics:**
- Throughput: Define as tokens processed per second during full task completion
- TTFT (Time to First Token): Measure in milliseconds from input submission to first output
- Model Parameter Efficiency: Calculate as (task success score / activated parameters)
- Resource Utilization: Measure as (success rate / total computation used)
- Utility: unify the metrics of SA and MAS with different metrics

**Experimental Controls**
- How to control MAS?
- Stanard Task solving 
  - Complexity?
- Robutness: Fault-Tolerance Tasks
- Scalability: Large-scale tasks
  - increase the number of agents
  - increase the size of the model, quantization or existing provided models.
- 


[1] [Survey on Evaluation of LLM-based Agents](http://arxiv.org/abs/2503.16416)
[2] Adaptive test case construction http://arxiv.org/abs/2503.13335, http://arxiv.org/abs/2407.08351
[3] [LLM Leaderboard](https://www.vellum.ai/llm-leaderboard)
[4] [Find open source llm](https://easyllm.site/static/models.html)
[5] [Qwen-Moe](https://qwenlm.github.io/blog/qwen-moe/)
[6] [Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient](https://arxiv.org/pdf/2502.05172)
[7] [OpenLLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)

# Discussion
- How to construct MAS?
- How to define the unified metrics? 