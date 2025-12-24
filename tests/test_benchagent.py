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

# 简单的日志设置
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    evaluator = "hotpotqa"
    
    # 配置 BenchAgent 参数
    agent_config = {
        # BenchAgent 不需要 agents_num/rounds_num
        "model_name": "gpt-4.1",   
        "manager_tools": ["python_interpreter"], 
        # "ALL" 会加载所有可用工具，这里指定常用搜索工具
        "search_tools": ["ALL"],
        "memory": None, # 可以测试 "melo_memory"
        "verbosity_level": 2, # 通常 2 就足够了，10 可能会非常冗长
        "evaluator": evaluator,
        "max_steps": 15,       # Manager 最大步数
        "search_max_steps": 10 # Search Agent 最大步数
    }
    
    # 生成带时间戳的日志文件名 - 使用正确的 agent 名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/bench_agent_simple_{timestamp}.log"
    
    print(f"Logging to: {log_file}")
    
    # 使用 asyncio.run 运行异步方法
    summary = BenchmarkRunner().run(
        benchmark_name=evaluator,
        agent_system="bench_agent",  # 指定 bench_agent
        agent_config=agent_config,
        limit=2,                     # 限制数量用于快速测试
        pass_at_k=1,                 # 除非测试稳定性，否则通常设为 1
        log_file=log_file,# 串行运行以调试日志
    )
    
    print("\n==== BenchAgent 简易测试结果 ====")
    print(summary)

