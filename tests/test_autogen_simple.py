import sys
import os
import asyncio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

if __name__ == "__main__":
    agent_config = {
        "model_name": "gpt-4.1",
        "manager_tools": ["python_interpreter", "final_answer"],
        "search_tools": ["search", "wikipedia"],      
        "num_rounds": 3,                                   
        "verbosity_level": 2,
        "evaluator": "math",   
        "memory": None,           
    }

    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name="math",       
        agent_system="autogen",           
        agent_config=agent_config,
        limit=1,                          
        pass_at_k=1,
        log_file="logs/autogen_test.log", 
    ))

    # 4. 输出结果
    print("\n" + "="*30)
    print("==== AutoGen (BenchAgent版) 测试结果 ====")
    print(f"Summary: {summary}")
    print("="*30)