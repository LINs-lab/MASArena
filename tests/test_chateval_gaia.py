import sys
import os
import asyncio
import logging
from datetime import datetime

# 保证可以import mas_arena
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

# 简单的日志设置，确保我们可以看到输出
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    evaluator = "gaia"
    # 配置 ChatEval NewCore（GAIA level1）
    agent_config = {
        "num_rounds": 3,
        "model_name": "gpt-4.1",
        "manager_tools": ["ALL"],
        "search_tools": ["ALL"],
        "memory": None,
        "verbosity_level": 100,
        "evaluator": evaluator,
        "max_steps": 15,
        "search_max_steps": 12,
    }

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/chateval_newcore_gaia_{timestamp}.log"

    print(f"Running ChatEval NewCore GAIA (level1) test... Log: {log_file}")

    # 使用 asyncio.run 运行异步方法
    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name=evaluator,
        agent_system="chateval_newcore",
        agent_config=agent_config,
        limit=0,  # 0 表示没有限制，跑完 level1 全部题目
        pass_at_k=1,
        log_file=log_file,
        data_path="data/gaia_validate_level3.jsonl",  # GAIA level1
        concurrency=10,
    ))

    print("\n==== ChatEval NewCore GAIA (level1) 测试结果 ====")
    print(summary)