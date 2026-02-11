# GAIA Jarvis 准确率偏低原因分析

基于 `logs/jarvis_gaia_20260131_161826.log` 和 `results/gaia_jarvis_20260131_161826.json` 的分析。

## 当前结果

- **总题数**: 86
- **正确**: 25
- **准确率**: ~29%

## 主要问题

### 1. sheet_extractor 工具未注册（已修复）

**现象**：日志中反复出现：
```
Failed to create tool sheet_extractor: Nullable argument 'feature_type' in function signature should have key 'nullable' set to True in inputs.
Unknown tool: sheet_extractor
```

**原因**：`sheet_extractor` 的 `inputs` 里 `feature_type` 未声明 `"nullable": True`，与框架校验规则冲突，导致工具创建失败，整个 benchmark 过程中该工具始终不可用。

**影响**：所有依赖 Excel/表格解析的任务无法使用该工具，只能靠代码执行或其它工具，容易失败或超步数。

**修复**：已在 `mas_arena/tools/sheet_extractor.py` 中为 `feature_type` 增加 `"nullable": True`，并在 `forward()` 中对 `None` 做默认处理（默认 `"formats"`）。

---

### 2. GAIA 题目附件文件缺失

**现象**：大量题目返回 "file not found"、"File missing"、"file not found" 等。

**原因**：题目中给出的路径形如：
`/workspace/project_multi_agents_benchmark/data/files/gaia/validate/<uuid>.xlsx`  
但 `data/files/gaia/` 下为空或缺少对应文件。

**说明**：GAIA 的题目附件需要单独下载。仓库中有 `data/download/download_gaia.py`，用于从 HuggingFace 下载 metadata 和题目相关文件。若未执行下载或未下载完整，所有“带附件”的题目都会因文件缺失而失败。

**建议**：
- 运行 GAIA benchmark 前先执行：
  ```bash
  python data/download/download_gaia.py  # 或按脚本说明下载 validation 集
  ```
- 确认 `data/files/gaia/validate/` 下存在题目所需的 xlsx/pdf/png 等文件。

---

### 3. 搜索/API 配额或限流

**现象**：预测中出现 "unable to access live web or GitHub search results due to usage limits"、"Search agent consistently failed to retrieve data"、"usage limit" 等。

**原因**：长时间、高并发跑 86 题时触达外部搜索/API 的配额或限流，导致部分题目无法拿到实时数据。

**影响**：依赖网页、GitHub、百科等在线信息的题目会退回“无法访问”或错误答案，拉低准确率。

**建议**：适当降低并发、增加请求间隔，或为搜索/API 配置更高配额/多 key 轮询。

---

### 4. Code Agent 步数上限

**现象**：部分题目答案为 "Code agent reached maximum steps (15) without completing the task"。

**原因**：Code Agent 最大步数设为 15，复杂多步推理或多次试错会提前用尽步数。

**影响**：需要多轮代码执行或复杂检索+代码的题目容易在中途被截断，无法给出最终答案。

**建议**：若可配置，可适当提高 Code Agent 的 `max_steps`（需权衡耗时与稳定性）。

---

### 5. 工具与文件类型不匹配

**现象**：日志中有：
```
Error executing tool inspect_file_as_csv: Unsupported file type. Supported: ['.csv', '.tsv', '.txt']
```

**原因**：Agent 对 xlsx 等文件调用了只支持 csv/tsv/txt 的 `inspect_file_as_csv`，导致报错。

**影响**：表格类题目若未用对工具（应用 sheet 相关工具或代码），会因工具报错而失败。

**说明**：修复 sheet_extractor 注册后，Agent 应能使用正确的表格工具，减少此类错误。

---

### 6. 异步资源清理告警（Event loop is closed）

**现象**：大量 `RuntimeError: Event loop is closed`，发生在 `AsyncClient.aclose()` 等异步清理时。

**原因**：主流程在 event loop 关闭后才做 httpx 等异步 client 的关闭，属于异步生命周期管理问题。

**影响**：主要影响日志与进程退出，一般不直接改变单题答案，但可能干扰并发与资源释放。

**建议**：在关闭 event loop 前确保所有 async client 已正确 `aclose()`，或在退出前统一做异步清理。

---

## 建议的修复与复测顺序

1. **已做**：修复 `sheet_extractor` 的 nullable 声明与 `forward()` 默认值。
2. **必须**：确保 GAIA 题目附件已完整下载到 `data/files/gaia/validate/`（及 test 如需要）。
3. **可选**：适当降低并发、提高搜索/API 配额或重试策略。
4. **可选**：适当提高 Code Agent `max_steps` 并复测易超步数的题目。
5. 用相同命令重新跑一遍 GAIA Jarvis，对比新结果与上述日志/结果文件。

完成 1 和 2 后，准确率应有明显提升；3–5 可进一步减少因环境和步数限制导致的失分。

---

## 日志 jarvis_gaia_20260131_211929 错误汇总

基于 `logs/jarvis_gaia_20260131_211929.log` 的逐条错误统计与修复建议。

### 错误类型统计

| 错误类型 | 出现次数 | 说明 |
|----------|----------|------|
| **inspect_file_as_csv: Unsupported file type** | 8+ | Agent 对 .png / .xlsx / .pdb 等调用了仅支持 .csv/.tsv/.txt 的工具 |
| **RuntimeError: Event loop is closed** | 数十次 | 每个 problem 结束后未显式 `agent.aclose()`，httpx AsyncClient 在 GC 时 aclose 时 event loop 已关闭 |
| **max_tokens / model output limit was reached** | 3 次 | 单次回复超过模型 max_completion_tokens，回复被截断 |
| **extract_zip_file: Unsupported file type** | 1 次 | 对非 ZIP 文件调用了 ZIP 解压工具 |
| **Execution timed out after 30 seconds** | 1+ | 代码执行 30 秒超时 |
| **FunctionTimedOut (search_agent 30s)** | 1 次 | 搜索或代码执行超时，伴随 Fatal write error on socket |
| **pdfminer gray color invalid float** | 大量 WARNING | PDF 解析兼容性警告，一般不影响主流程 |

### 已做修复

1. **Event loop is closed**
   - 在 `benchmark_runner._process_one_problem` 的 `finally` 中增加 `await agent.aclose()`，在同一 event loop 内关闭 httpx 等资源。
   - 在 `_process_one_problem_pass_at_k` 的 `run_one_sample()` 内对每个 agent 在 `evaluate` 后同样 `aclose()`。

2. **inspect_file_as_csv 误用**
   - 在 `mas_arena/tools/csv_extractor.py` 中强化工具描述：明确仅支持 .csv/.tsv/.txt，并写明“Excel 用 extract_sheet_features，图片用 inspect_file_as_image”等，减少误用。

### 建议后续优化

1. **max_tokens 不足**  
   若仍出现 “max_tokens or model output limit was reached”，可在创建模型时提高 `max_completion_tokens`（例如 8192→16384）。BenchAgent 已用 8192；若 Jarvis 走 agent_core 且默认 4096，可在配置或 `OpenAIServerModel` 中提高。

2. **工具路由**  
   可在 `tool_manager` 中根据 `file_path` 后缀做预检查：若扩展名不在工具支持列表内，直接返回明确错误和应使用的工具名，减少无效调用。

3. **代码/搜索超时**  
   30 秒代码超时、search_agent FunctionTimedOut 可按需适当提高 timeout 或增加重试，需权衡总耗时。
