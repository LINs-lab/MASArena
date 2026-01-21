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
    # 配置 LLM Debate 参数
    agent_config = {
        "agents_num": 2,            # 2个辩手
        "rounds_num": 2,            # 2轮辩论
        "model_name": "gpt-4.1",   # 模型
        "manager_tools": ["python_interpreter"], # final_answer 现在会自动添加
        "search_tools": ["ALL"],    # 使用全部工具
        "memory": None,
        "verbosity_level": 1,
        "evaluator": evaluator,        
    }
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/llm_debate_gaia_{timestamp}.log"
    
    print(f"Running LLM Debate GAIA test... Log: {log_file}")
    
    # 使用 asyncio.run 运行异步方法
    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name=evaluator,
        agent_system="llm_debate",  # 指定 llm_debate 系统
        agent_config=agent_config,
        limit=1,                    
        pass_at_k=1,
        log_file=log_file,
    ))
    
    print("\n==== LLM Debate GAIA 测试结果 ====")
    print(summary)

