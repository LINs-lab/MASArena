# 完整 Prompt 索引

本目录包含所有 New Core MAS 的 **完整 prompt 原文**（非摘要）。

## 共享层 (BenchAgent)
- `shared/prompts.yaml` — base_manager_prompt, search_agent_prompt_managed_task 等
- `shared/format_prompts.py` — 各 benchmark format prompt 源码
- `shared/gaia_format_prompt.txt` — GAIA 完整 format prompt
- `shared/run_agent_user_prefix.txt` — Jarvis 等 run_agent 模式的 user 前缀
- `shared/codeagent_default_system_template.txt` — Manager CodeAgent 默认 system
- `shared/toolcalling_default_system_template.txt` — Search ToolCallingAgent 默认 system
- `shared/manager_instructions.txt` — CodeAgent instructions 字段
- `shared/search_agent_json_suffix.txt` — Search Agent JSON 格式后缀

## 各 MAS
- `chateval/` — 3 角色 role prompt + debate context + synthesis + rescue
- `jarvis/` — Jarvis additional_instructions
- `llm_debate/` — 首轮/辩论/聚合 prompt 模板
- `autogen/` — Primary/Critic 完整 prompt
- `camel/` — User/Assistant/Summary 完整 prompt
- `swarm/` — 求解 + 聚合 prompt
- `evoagent/` — 初始 3 模板 + crossover/mutation/summary meta prompt
- `metagpt/` — 5 角色 system prompt 原文 + Engineer BenchAgent 模板
