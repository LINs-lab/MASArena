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
    # 配置 Jarvis 参数
    agent_config = {
        "model_name": "gpt-4.1",   # 模型
        "manager_tools": ["ALL"], 
        "search_tools": ["ALL"], # 显式添加 crawler_read 以启用 Jina
        "memory": None,
        "verbosity_level": 100, 
        "evaluator": evaluator,        
    }
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/jarvis_gaia_{timestamp}.log"
    
    print(f"Running Jarvis GAIA test... Log: {log_file}")

    # 使用 asyncio.run 运行异步方法
    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name=evaluator,
        agent_system="jarvis",  # 指定 jarvis 系统
        agent_config=agent_config,
        limit=0,  # 0 表示没有限制，运行 gaia level1 的所有题目                    
        pass_at_k=1,
        log_file=log_file,
        data_path="data/gaia_validate_level1.jsonl", # 显式指定数据文件
        concurrency=10, # 增加并发
    ))
    
    print("\n==== Jarvis GAIA 测试结果 ====")
    print(summary)

