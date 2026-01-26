import sys
import os
import asyncio
import logging
from datetime import datetime

# Ensure we can import mas_arena
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mas_arena.benchmark_runner import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    evaluator = "gaia"
    
    # Configure the agent with ALL tools enabled
    # This configuration relies on the fix applied to BenchAgent to read tools from config
    agent_config = {
        "model_name": "gpt-4.1",  # Use a strong model for complex tasks
        "manager_tools": ["ALL"], # Enable all tools (Visual, Search, etc.)
        "search_tools": ["ALL"],
        "memory": None,
        "verbosity_level": 100, 
        "evaluator": evaluator,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/bench_agent_gaia_10_{timestamp}.log"
    
    print(f"Running GAIA benchmark (10 problems, concurrent)... Log: {log_file}")
    print(f"Agent Config: {agent_config}")

    try:
        summary = asyncio.run(BenchmarkRunner().arun(
            benchmark_name=evaluator,
            agent_system="bench_agent",  # Use bench_agent to ensure our tool config is used
            agent_config=agent_config,
            limit=1,        # Limit to 10 problems
            concurrency=10,  # Run 10 problems concurrently
            pass_at_k=1,
            log_file=log_file,
            data_path="data/gaia_validate_level1.jsonl",
        ))
        
        print("\n==== Test Results ====")
        print(summary)
        
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()

