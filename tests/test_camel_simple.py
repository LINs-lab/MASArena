import sys
import os
import asyncio

# 保证可以 import mas_arena 以及项目中的其他模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

async def main():
    agent_config = {
        "model_name": "gpt-4.1",
        "manager_tools": ["python_interpreter", "final_answer"],
        "search_tools": ["search", "wikipedia"],
        "verbosity_level": 2,
        "evaluator": "math",
        "memory": None,
    }


    summary = await BenchmarkRunner().arun(
        benchmark_name="math", 
        agent_system="camel",    
        agent_config=agent_config,
        limit=1,              
        pass_at_k=1,
        log_file="logs/camel_test.log",
    )

    print("\n" + "="*20 + " Camel 简易测试结果 " + "="*20)
    print(summary)

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未检测到 OPENAI_API_KEY 环境变量")
        
    asyncio.run(main())