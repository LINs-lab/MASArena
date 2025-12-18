import sys
import os
import asyncio

# 保证可以从项目根目录 import mas_arena
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

async def main():

    agent_config = {
        "model_name": "gpt-4.1",
        "search_max_steps": 10,
        "max_iterations": 3,
        "verbosity_level": 2,
        "manager_tools": ["python_interpreter", "final_answer"],
        "search_tools": ["search", "wikipedia"],
        "evaluator": "math",
    }

    runner = BenchmarkRunner()
    
    print("==== 开始 MetaGPT 基准测试 ====")
    
    summary = await runner.arun(
        benchmark_name="math",      
        agent_system="metagpt",     
        agent_config=agent_config,
        limit=1,                    
        pass_at_k=1,
        log_file="logs/metagpt_test.log",
    )

    print("\n" + "="*30)
    print("==== MetaGPT 测试完成 ====")
    print(summary)
    print("="*30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")