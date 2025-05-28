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



## Meeting notes

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

Feishu Page: [project_multi_agents_benchmark](https://ocnfww8fyyv6.feishu.cn/docx/PVXfdIcvYof6R8xKon4cCyQJncb)

