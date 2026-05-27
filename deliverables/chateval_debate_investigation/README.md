# ChatEval 辩论内容调查 & New Core MAS Prompt 结构

生成时间：2026-05-25

## 一、ChatEval 详细 debate 内容在哪里？

### 结论

`results/agent_responses/` 里的 ChatEval **确实不能完整反映 agents 之间的辩论轮次**，原因在代码层面：

1. **`chateval_newcore.py`** 在内存中维护 `debate_history: List[str]`（如 `"Math Expert (Round 1): ..."`），但 `base.py` 的 `_record_agent_responses()` **只序列化 `run_output["messages"]`**，不会单独写入 `debate_history`。
2. **`messages` 来自各 BenchAgent 的 `run_agent_step()`**，典型结构是：
   - `user`：完整 prompt，内含累积的 `Discussion History:`（辩论摘要）
   - `assistant` / `name: bench_agent_final`：该轮 **final_answer**（通常很短）
3. 因此 JSON 里 assistant 条目看起来像「只有最终答案」；**完整轮次藏在 user 消息的 `Discussion History` 段**，或只在 **logs** 里以 `********` messages dump / `STEP_LOG` 形式出现。

### 非 Qwen 记录位置（2026-01 ~ 2026-02，均在 2025 年 5 月之后）

| 类型 | 路径模式 | 说明 |
|------|----------|------|
| GAIA 主跑 | `logs/chateval_newcore_gaia_202602*.log` | 35 个文件 |
| GAIA nohup | `logs/chateval_newcore_gaia_nohup/chateval_newcore_gaia_202602*.log` | 后台运行副本 |
| GAIA search 专项 | `logs/chateval_newcore_gaia_search_202602*.log` | 搜索相关 |
| 其他 benchmark | `logs/chateval_newcore_{bbh,gsm8k,...}_202601*.log` | MBPP/MMLU/BBH 等 |

**已排除（Qwen）：**
- `logs/chateval_gaia/chateval_gaia_qwen3-32b_*.log`
- `logs/chateval_newcore_gaia/chateval_newcore_gaia_qwen3-32b_*.log`

**当前工作区 `results/agent_responses/` 为空**（被 `.gitignore` 忽略；日志中有 `Saved agent responses to ...` 说明运行时曾生成，但本地未保留）。

### 日志中的辩论内容形态

1. **`Discussion History:` 段**（嵌入 user prompt）
   ```
   Discussion History:
   Math Expert (Round 1): Guava
   Logic Expert (Round 1): Guava
   ...
   ```

2. **`debate_history` 字符串**（runtime，`run_output` 返回但 **未写入 agent_responses**）
   ```python
   debate_history.append(f"{agent_name} (Round {round_idx + 1}): {response_text}")
   ```

3. **`********` 块** — `base.evaluate()` 打印的完整 messages list

4. **`STEP_LOG`** — BenchAgent 工具调用与推理步骤（最细粒度）

### 本包附带内容

- `full_prompts/` — 所有 New Core MAS 完整 prompt 原文（见 `full_prompts_INDEX.md`）
- `extract_chateval_debates.py` — 辩论内容提取脚本
- `export_full_prompts.py` — 完整 prompt 导出脚本
- `extracted_debates/` — 整理后的辩论数据：
  - `by_log/` — 每个 log 一个 JSON（按题目分组）
  - `transcripts/` — 每题一个 Markdown 可读 transcript
  - `consolidated/all_debates.json` — 全部辩论合并 JSON
  - `README.md` / `_summary.json` — 索引与统计
- `source/` — 相关源码副本

### 如何恢复完整 transcript

```bash
python3 extract_chateval_debates.py
# 或 grep 单文件
grep -o 'Math Expert (Round [0-9]*):[^\\n]*' logs/chateval_newcore_gaia_20260209_195838.log
```

---

## 二、New Core MAS Prompt 结构总览

所有 `*_newcore.py` 共享 **BenchAgent** 执行引擎（`bench_agent.py` + `prompts.yaml` + `format_prompts.py`）。

### 共享 BenchAgent 三层 Prompt

```
Manager (CodeAgent) system =
  agent_core 默认 ReAct 模板
  + prompts.yaml → base_manager_prompt
  + additional_instructions（各 MAS 注入）
  + manager_instructions（可委托 search_agent）

Search (ToolCallingAgent) system =
  默认模板 + JSON tool-call 格式
  + search_agent_prompt_managed_task

运行时 user prompt (run_agent) =
  "Answer this question correctly..." 前缀
  + 题目 + 附件描述
  + format_prompt（按 benchmark）
  + [可选] 记忆检索 / 语义校验 / 失败重试

run_agent_step（ChatEval/Debate 等单步模式）=
  augmented_question（MAS 构造）
  + format_prompt 追加在末尾
```

详见 `prompt_structures.md` 与各 `source/*.py`。

### 各 MAS 差异速查

| MAS | 编排 | Prompt 注入点 | 最终答案 |
|-----|------|---------------|----------|
| **ChatEval** | 3 BenchAgent 轮询 × N 轮 | `role_prompt` + `_build_context()` | LLM 合成器 `_extract_final_result()` |
| **Jarvis** | 单 BenchAgent ReAct | `jarvis_instructions` → `additional_instructions` | BenchAgent final_answer |
| **LLM Debate** | N BenchAgent 并行 + 聚合 | 首轮/辩论/聚合 prompt | Aggregator BenchAgent |
| **AutoGen** | Primary(BenchAgent) + Critic(API) | 内联 system prompt + 对话历史 | Primary 最后输出 |
| **CAMEL** | User/Assistant API + 总结 | 角色 system prompt | BenchAgent 总结 step |
| **Swarm** | 并行 BenchAgent + 聚合 | `_create_prompt()` / `aggregate()` | Aggregator |
| **EvoAgent** | 进化 + BenchAgent 求解 | 可变 `system_prompt` + meta-prompts | LLM 汇总 |
| **MetaGPT** | 5 角色流水线 | 各角色长 system prompt | QA Engineer |

---

## 三、源码参考路径

```
mas_arena/agents/chateval_newcore.py
mas_arena/agents/jarvis_newcore.py
mas_arena/agents/llm_debate_newcore.py
mas_arena/agents/autogen_newcore.py
mas_arena/agents/camel_newcore.py
mas_arena/agents/swarm_newcore.py
mas_arena/agents/evoagent_newcore.py
mas_arena/agents/metagpt_newcore.py
mas_arena/agents/bench_agent.py
mas_arena/agents/prompts.yaml
mas_arena/agents/format_prompts.py
mas_arena/agents/base.py  (_record_agent_responses)
```
