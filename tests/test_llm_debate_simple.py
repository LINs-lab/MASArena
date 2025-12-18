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
    # 配置 LLM Debate 参数
    agent_config = {
        "agents_num": 2,            # 2个辩手
        "rounds_num": 2,            # 2轮辩论
        "model_name": "gpt-4o-mini",   # 模型
        "manager_tools": ["python_interpreter"], # final_answer 现在会自动添加
        "search_tools": ["search", "wikipedia"],
        "memory": None,
        "verbosity_level": 2,
        "evaluator": "math",        # 使用 Math 评估器
    }
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/llm_debate_simple_{timestamp}.log"
    
    # 使用 asyncio.run 运行异步方法
    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name="math",
        agent_system="llm_debate",  # 指定 llm_debate 系统
        agent_config=agent_config,
        limit=1,                    # 限制只跑2个问题用于快速测试
        pass_at_k=1,
        log_file=log_file,
    ))
    
    print("\n==== LLM Debate 简易测试结果 ====")
    print(summary)
