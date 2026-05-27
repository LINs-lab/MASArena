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
    evaluator = "mbpp" # Using a simple evaluator
    # Configure ChatEval NewCore
    agent_config = {
        "num_rounds": 3,
        "model_name": "gpt-4.1",   # Using a cheaper model for test
        "manager_tools": ["python_interpreter"], # Give them python
        "search_tools": ["ALL"], # No search needed for simple math
        "memory": None,
        "verbosity_level": 10,
        "evaluator": evaluator,
        "max_steps": 15,
        "search_max_steps": 10,
    }
    
    # Generate log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/chateval_newcore_{evaluator}_{timestamp}.log"
    
    print(f"Running ChatEval NewCore test... Log: {log_file}")
    
    # Use asyncio.run 
    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name=evaluator,
        agent_system="chateval_newcore",  # Specify our new agent
        agent_config=agent_config,
        limit=400, # Run just 1 problem
        pass_at_k=1,
        log_file=log_file,
    ))
    
    print("\n==== ChatEval NewCore Test Results ====")
    print(summary)

