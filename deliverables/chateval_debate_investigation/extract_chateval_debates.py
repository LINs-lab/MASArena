#!/usr/bin/env python3
"""从 ChatEval NewCore 日志提取辩论内容（非 Qwen），按题目整理并生成 transcript。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "logs"
BASE_OUT = Path(__file__).resolve().parent / "extracted_debates"

AGENTS = ("Math Expert", "Logic Expert", "Critical Thinking Expert")
AGENT_ORDER = {"Math Expert": 0, "Logic Expert": 1, "Critical Thinking Expert": 2}
AGENT_SPLIT_RE = re.compile(
    r"(Math Expert \(Round \d+\):|Logic Expert \(Round \d+\):|Critical Thinking Expert \(Round \d+\):)"
)
HEADER_RE = re.compile(
    r"^(Math Expert|Logic Expert|Critical Thinking Expert) \(Round (\d+)\):"
)
PROBLEM_RE = re.compile(
    r"Original Problem:\s*(.*?)(?:\\n\\n|\n\n)Discussion History:",
    re.DOTALL,
)
TRUNCATE_MARKERS = (
    "Math Expert (Round",
    "Logic Expert (Round",
    "Critical Thinking Expert (Round",
    "\\n\\nIt is now Round",
    "\\n\\nYou are an all-capable",
    "\\n\\n📋 OUTPUT FORMAT",
    "\\n\\n⚠️ FILE PROCESSING",
)


def is_qwen_log(path: Path) -> bool:
    return "qwen" in path.name.lower()


def find_logs() -> list[Path]:
    return sorted(
        {p for p in LOGS_DIR.rglob("chateval*.log") if p.is_file() and not is_qwen_log(p)}
    )


def _clean(s: str) -> str:
    s = s.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _clean_one_line(s: str) -> str:
    return re.sub(r"\s+", " ", _clean(s).replace("\n", " ")).strip()


def _slug(text: str, n: int = 48) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text[:n]).strip("_") or "problem"
    return f"{safe}_{h}"


def extract_rounds_from_segment(segment: str) -> list[dict]:
    debate: dict[tuple[str, int], dict] = {}
    parts = AGENT_SPLIT_RE.split(segment)
    i = 1
    while i + 1 < len(parts):
        header = parts[i].strip()
        body = parts[i + 1]
        hm = HEADER_RE.match(header)
        if hm:
            agent, rnd = hm.group(1), int(hm.group(2))
            for marker in TRUNCATE_MARKERS:
                pos = body.find(marker)
                if pos > 0:
                    body = body[:pos]
            content = _clean_one_line(body)
            if content and not content.startswith("[Error]"):
                key = (agent, rnd)
                if key not in debate or len(content) > len(debate[key]["content"]):
                    debate[key] = {"agent": agent, "round": rnd, "content": content}
        i += 2
    return sorted(
        debate.values(),
        key=lambda x: (x["round"], AGENT_ORDER.get(x["agent"], 99), x["agent"]),
    )


def extract_problems_from_blob(blob: str) -> list[dict]:
    """按 Original Problem 切段，合并同题各段的 debate rounds（取最完整内容）。"""
    segments = re.split(r"(?=Original Problem:)", blob)
    by_problem: dict[str, dict[tuple[str, int], dict]] = {}

    for seg in segments:
        if "Discussion History:" not in seg:
            continue
        pm = PROBLEM_RE.search(seg)
        if not pm:
            continue
        problem = _clean_one_line(pm.group(1))
        if not problem:
            continue
        rounds = extract_rounds_from_segment(seg)
        if not rounds:
            continue
        store = by_problem.setdefault(problem, {})
        for r in rounds:
            key = (r["agent"], r["round"])
            if key not in store or len(r["content"]) > len(store[key]["content"]):
                store[key] = r

    results: list[dict] = []
    for problem, store in by_problem.items():
        rounds = sorted(
            store.values(),
            key=lambda x: (x["round"], AGENT_ORDER.get(x["agent"], 99), x["agent"]),
        )
        results.append(
            {
                "problem": problem,
                "debate_round_count": len(rounds),
                "debate_rounds": rounds,
            }
        )
    return results


def process_log(path: Path) -> dict:
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Discussion History:" in line or any(a in line for a in AGENTS):
                chunks.append(line)
    blob = "\n".join(chunks)
    problems = extract_problems_from_blob(blob)
    return {
        "log_file": str(path.relative_to(ROOT)),
        "log_name": path.stem,
        "problem_count": len(problems),
        "problems": problems,
    }


def render_transcript(problem: str, rounds: list[dict]) -> str:
    lines = [
        "# ChatEval Debate Transcript",
        "",
        "## Original Problem",
        "",
        problem,
        "",
        "## Debate Rounds",
        "",
    ]
    current_round = 0
    for r in rounds:
        if r["round"] != current_round:
            current_round = r["round"]
            lines.append(f"### Round {current_round}")
            lines.append("")
        lines.append(f"**{r['agent']}**")
        lines.append("")
        lines.append(r["content"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_transcripts(log_data: dict, transcript_root: Path) -> None:
    log_dir = transcript_root / log_data["log_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(log_data["problems"], 1):
        slug = _slug(item["problem"])
        md_path = log_dir / f"{idx:03d}_{slug}.md"
        md_path.write_text(
            render_transcript(item["problem"], item["debate_rounds"]),
            encoding="utf-8",
        )


def main() -> None:
    for sub in ("by_log", "transcripts", "consolidated"):
        p = BASE_OUT / sub
        if p.exists():
            for f in p.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    import shutil

                    shutil.rmtree(f)
        p.mkdir(parents=True, exist_ok=True)

    logs = find_logs()
    log_summaries = []
    all_problems: list[dict] = []

    for log in logs:
        try:
            data = process_log(log)
            out_path = BASE_OUT / "by_log" / f"{log.stem}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            if data["problem_count"] > 0:
                write_transcripts(data, BASE_OUT / "transcripts")
                for p in data["problems"]:
                    all_problems.append(
                        {
                            "log_file": data["log_file"],
                            "log_name": data["log_name"],
                            **p,
                        }
                    )

            log_summaries.append(
                {
                    "log_file": data["log_file"],
                    "problem_count": data["problem_count"],
                    "total_rounds": sum(p["debate_round_count"] for p in data["problems"]),
                }
            )
        except Exception as e:
            log_summaries.append(
                {"log_file": str(log.relative_to(ROOT)), "error": str(e)}
            )

    (BASE_OUT / "consolidated" / "all_debates.json").write_text(
        json.dumps(all_problems, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with_debate = [s for s in log_summaries if s.get("problem_count", 0) > 0]
    (BASE_OUT / "_summary.json").write_text(
        json.dumps(
            {
                "total_logs": len(logs),
                "logs_with_debate": len(with_debate),
                "total_problems": len(all_problems),
                "logs": log_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (BASE_OUT / "_log_index.txt").write_text(
        "\n".join(str(p.relative_to(ROOT)) for p in logs), encoding="utf-8"
    )

    readme = f"""# ChatEval 辩论内容提取结果（非 Qwen）

- 扫描日志: {len(logs)} 个
- 含辩论内容的日志: {len(with_debate)} 个
- 提取到的题目/辩论会话: {len(all_problems)} 个

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
{{
  "problem": "原题文本",
  "debate_round_count": 6,
  "debate_rounds": [
    {{"agent": "Math Expert", "round": 1, "content": "..."}},
    ...
  ]
}}
```

## 说明

- 来源: `logs/chateval*.log`（排除 qwen）
- `agent_responses` 通常只有 final_answer；完整轮次来自 log 中 `Discussion History:` 段
- 早期部分 benchmark log 无 Discussion History 标记，因此 debate 为空
"""
    (BASE_OUT / "README.md").write_text(readme, encoding="utf-8")

    print(
        f"Done: {len(logs)} logs, {len(with_debate)} with debate, "
        f"{len(all_problems)} problems -> {BASE_OUT}"
    )


if __name__ == "__main__":
    main()
