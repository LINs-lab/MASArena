# ChatEval 辩论内容提取结果（非 Qwen）

- 扫描日志: 64 个
- 含辩论内容的日志: 14 个
- 提取到的题目/辩论会话: 1986 个

## 目录说明

| 目录/文件 | 说明 |
|-----------|------|
| `by_log/` | 每个 log 一个 JSON，内含该 log 下所有题目的 debate_rounds |
| `transcripts/` | 按 log 分目录，每题一个 Markdown 可读 transcript |
| `consolidated/all_debates.json` | 全部题目辩论的合并 JSON |
| `_summary.json` | 汇总索引 |
| `_log_index.txt` | 非 Qwen 日志文件列表 |

## 数据字段

```json
{
  "problem": "原题文本",
  "debate_round_count": 6,
  "debate_rounds": [
    {"agent": "Math Expert", "round": 1, "content": "..."},
    ...
  ]
}
```

## 说明

- 来源: `logs/chateval*.log`（排除 qwen）
- `agent_responses` 通常只有 final_answer；完整轮次来自 log 中 `Discussion History:` 段
- 早期部分 benchmark log 无 Discussion History 标记，因此 debate 为空
