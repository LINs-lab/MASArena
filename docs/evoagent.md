# EvoAgent 交接文档

## 概览

EvoAgent 是基于进化算法的多智能体系统，通过三轮迭代（初始化 → 交叉 → 变异）筛选出最优 agent 组合，再由汇总 LLM 聚合最终答案。

主要文件：`mas_arena/agents/evoagent_newcore.py`

---

## 算法流程

```
初始化 3 个 base agents（不同人设的系统提示）
         ↓
并发执行，各自用 BenchAgent 求解问题
         ↓
第一轮迭代（交叉）：保留 best base agent + 生成 5 个交叉后代
         ↓
并发执行交叉 agents
         ↓
第二轮迭代（变异）：保留 best crossover agent + 生成 8 个变异后代
         ↓
并发执行变异 agents
         ↓
按 evaluator 评分排序，取 top-5 final agents
         ↓
_summarize_results：用一个独立 LLM 整合 5 个答案 → final_answer
```

各阶段 agent 数量（`initial_agents_count=3`）：
- base: 3
- crossover: 1 + (3×2−1) = 6
- mutation: 1 + (3×3−1) = 9
- final: top 5（从 mutation_agents 里取）

---

## 核心类

### `BenchEnhancedAgent`（dataclass）

单个进化 agent，内部持有一个 `BenchAgent` 实例负责实际求解。

| 字段 | 说明 |
|------|------|
| `agent_id` | UUID |
| `name` | 显示名，如 `EVO-1`、`EVO-C-3`、`EVO-M-7` |
| `system_prompt` | 该 agent 的人设提示 |
| `score` | evaluator 打分（0.0 ~ 1.0） |
| `result` | `solve()` 返回的 dict，含 `extracted_answer`、`usage_metadata` 等 |
| `bench_agent` | 实际执行的 `BenchAgent`，base agents 共享 `EvoAgent.bench_agent_executor` |

`solve(problem)` → 调用 `bench_agent.run_agent_step` → 从返回消息中提取 token 用量 → 返回 `result` dict。

### `EvoAgent`（继承 `AgentSystem`）

| 方法 | 作用 |
|------|------|
| `run_agent(problem)` | 主入口，执行完整进化流程，返回标准 `run_output` dict |
| `_initialize_base_agents()` | 创建 3 个初始 agent，共享 `bench_agent_executor` |
| `_crossover(p1, p2)` | 用 LLM 合并两个 agent 的系统提示 |
| `_mutation(parent)` | 用 LLM 对一个 agent 的系统提示做变异 |
| `_run_agent_task(agent, ...)` | 并发执行单个 agent，捕获超时/异常，写回 `agent.score` 和 `agent.result` |
| `_calculate_score(result, problem)` | 调用 `self.evaluator.calculate_score` 对 agent 答案打分 |
| `_summarize_results(problem, results)` | 用独立 LLM 聚合 final agents 的答案，返回 `(summary_text, usage_metadata)` |

---

## `run_agent` 返回结构

```python
{
    "messages": [
        HumanMessage(content=problem_text),
        AIMessage(content=agent.result["extracted_answer"], name="EVO-x", usage_metadata={...}),
        # × final_agents_count 条
        AIMessage(content=summary, name="EVO-SUMMARY", usage_metadata={...}),
    ],
    "final_answer": summary,          # 供 evaluator 使用
    "execution_time_ms": float,
    "evolution_metrics": {
        "initial_agents": int,
        "crossover_agents": int,
        "mutation_agents": int,
        "final_agents": int,
        "best_score": float,
    }
}
```

`EVO-SUMMARY` 的 `usage_metadata` 只包含 non-final-agents（base + crossover + mutation 中不在 final 里的）的 token 合计 + 汇总 LLM 自身的 token，**不包含** final agents（它们各自的 AIMessage 里已有）。这样 `base.py._record_token_usage` 汇总后不会重复计数。

---

## 配置参数

通过 `agent_config` dict 传入（`main.py` 或 `BenchmarkRunner.arun`）：

| 参数 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| `model_name` | `MODEL_NAME` | `gpt-4o-mini` | 所有 LLM 共用 |
| `initial_agents_count` | — | `3` | base agent 数 |
| `final_agents_count` | — | `5` | 最终参与汇总的 agent 数 |
| `agent_task_timeout_seconds` | `MAS_ARENA_EVO_AGENT_TIMEOUT_SECONDS` | `300` | 单个 agent 求解超时 |
| `evolution_step_timeout_seconds` | `MAS_ARENA_EVO_STEP_TIMEOUT_SECONDS` | `60` | crossover/mutation LLM 调用超时 |
| `summary_timeout_seconds` | `MAS_ARENA_EVO_SUMMARY_TIMEOUT_SECONDS` | `300` | 汇总 LLM 超时 |
| `manager_tools` | — | `None` | `BenchAgent` 可用的管理工具，`"ALL"` 开全部 |
| `search_tools` | — | `None` | `BenchAgent` 可用的搜索工具，`"ALL"` 开全部 |
| `search_max_steps` | — | `10` | `BenchAgent` 最大搜索步数 |

---

## 运行方式

### 全量评测

```bash
python main.py \
  --benchmark gaia \
  --agent-system evoagent \
  --data data/gaia_validate_level2.jsonl \
  --model-name qwen3-32b \
  --manager-tools ALL \
  --search-tools ALL \
  --concurrency 13 \
  --results-dir results
```

`data/gaia_validate_level2_remaining.jsonl`：已跑完的 13 题去掉后的 73 题，可直接替换 `--data` 续跑。

### 调试单题

```bash
python scripts/debug_evoagent_one.py \
  --data data/gaia_validate_level2.jsonl \
  --data-id df6561b2-7ee5-4540-baab-5095f742716a \
  --model-name qwen3-32b \
  --manager-tools ALL \
  --search-tools ALL \
  --break-after-run-agent   # 在 run_agent 返回后进 pdb
```

`--no-evaluate` 可跳过 evaluator，只看 agent 原始输出。

VSCode 调试配置在 `.vscode/launch.json`，三个 configuration 对应上面两种场景。

---

## 并发运行与 Browser Pool

### 问题背景

高并发（如 13）时，每个 `BenchAgent` 会各自初始化一个 `Browser` 实例，导致启动 N 个独立 Chromium 进程，内存直接爆炸（每进程约 500 MB）。

### 解决方案：`_BrowserPool`（`mas_arena/tools_old/browser_tool.py`）

进程级单例，整个进程只启动一个 Chromium，所有并发 worker 共享，通过 semaphore 控制同时活跃的 browser context 数量。

```
并发 worker × N
     │
     ▼
_BrowserPool（单例）
  ├─ 1 个 Playwright 实例
  ├─ 1 个 Chromium 进程
  └─ semaphore(POOL_SIZE) ← 限制同时活跃的 context 数
         │
  acquire_context() → 借出 context + page
  release_context() → 关闭 context，释放 slot
```

**环境变量：**

| 变量 | 默认 | 说明 |
|------|------|------|
| `MAS_ARENA_BROWSER_POOL_SIZE` | `4` | 最大同时活跃 context 数，超出的 worker 排队等待 |
| `MAS_ARENA_BROWSER_ACQUIRE_TIMEOUT_SECONDS` | `300` | 等待 slot 的超时，超时抛 `RuntimeError` 而非永久阻塞 |

**实测（13 并发，POOL_SIZE=4，约 11 分钟）：**
- Chromium 进程数全程 = 1（修复前 = 13）
- 内存峰值约 1.9 GB（修复前估算 7+ GB）
- 内存全程平坦，无泄漏，进程正常退出

### `benchmark_runner.py` 配套修复

正常评测路径（非 pass_at_k）的 `process_with_semaphore` 之前不调用 `agent.aclose()`，导致 `Browser.close()` 从未执行，pool slot 永远不释放。已在 `finally` 块中补上：

```python
async def process_with_semaphore(i, p):
    async with semaphore:
        agent = create_agent_system(...)
        try:
            return await self._process_one_problem(i, p, agent, ...)
        finally:
            if hasattr(agent, "aclose"):
                await agent.aclose()
```

`bench_agent.aclose()` 会遍历 `search_tools` / `manager_tools`，对每个有 `browser_instance` 的工具调用 `browser_instance.close()`，即 `pool.release_context()`。

```
BenchmarkRunner.arun()
  └─ _process_one_problem()
       └─ agent.evaluate()          # base.py AgentSystem.evaluate
            ├─ run_agent()          # EvoAgent 实现
            ├─ _record_token_usage(messages)
            └─ evaluator.evaluate(problem, run_result=run_output)
                 └─ gaia_evaluator: 从 run_result["final_answer"] 提取答案
```

`format_prompt`（GAIA 要求 `<answer>...</answer>` 格式）在 `base.py.__init__` 时依据 `evaluator_name` 初始化。`_summarize_results` 的提示词末尾嵌入 `self.format_prompt`，引导汇总 LLM 输出标准格式。

---

## 已知问题与修复记录

| 日期 | 文件 | 问题 | 修复 |
|------|------|------|------|
| 2026-05-23 | `base.py:208` | `usage_metadata["output_token_details"]` 硬访问，EVO-SUMMARY 消息缺该 key 导致 `KeyError`，整条评测链崩溃，`prediction` 永远为空 | 改为 `.get("output_token_details", {})` |
| 2026-05-23 | `evoagent_newcore.py:117` | `BenchAgent._extract_conversation_history` 返回的 `usage_metadata` 是 `CompletionUsage` 对象，但 `solve()` 只检查 `isinstance(um, dict)`，token 永远为 0 | 增加 `isinstance(um, CompletionUsage)` 分支 |
| 2026-05-23 | `evoagent_newcore.py:767` | `_all_agents` 聚合用了 `getattr(self, '_crossover_agents', [])` 但该属性从未赋值；且 `final_agents` 被重复计数 | 改用局部变量 `crossover_agents + mutation_agents`，排除 `final_agents` |
| 2026-05-24 | `tools_old/browser_tool.py` | 13 并发启动 13 个 Chromium 进程，服务器内存耗尽卡死 | 引入 `_BrowserPool` 进程级单例，所有 worker 共享 1 个 Chromium，semaphore 限制最大 context 数（默认 4） |
| 2026-05-24 | `tools_old/browser_tool.py` | `_BrowserPool.acquire_context()` 无超时，slot 泄漏时永久阻塞 | 改用 `asyncio.wait_for(semaphore.acquire(), timeout=300s)`，超时抛 `RuntimeError` |
| 2026-05-24 | `benchmark_runner.py` | 正常评测路径 `process_with_semaphore` 不调用 `agent.aclose()`，browser context slot 永不归还 | 在 `finally` 块补调 `agent.aclose()` |
