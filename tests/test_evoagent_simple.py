import sys
import os
import asyncio

# 1. 保证可以从项目根目录 import mas_arena
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

if __name__ == "__main__":

    agent_config = {
        "model_name": "gpt-4.1",      
        "initial_agents_count": 3,        
        "final_agents_count": 5,          
        "crossover_rate": 0.7,            
        "mutation_rate": 0.3,             
        "verbosity_level": 2,             
        "evaluator": "math",
        "manager_tools": ["python_interpreter", "final_answer"],
        "search_tools": ["search", "wikipedia"],
        "memory": None,              
    }

    summary = asyncio.run(BenchmarkRunner().arun(
        benchmark_name="math",          
        agent_system="evoagent",         
        agent_config=agent_config,       
        limit=1,                         
        pass_at_k=1,                     
        log_file="logs/evoagent_test.log" 
    ))

    # 4. 输出结果
    print("\n" + "="*20 + " EvoAgent 进化测试结果 " + "="*20)
    print(summary)