# test.py
import asyncio
import os
os.environ["CHROMA_TELEMETRY"] = "false"  # 关闭 Chroma 遥测

async def main():
    from mas_arena.agents.bench_agent import BenchAgent

    agent = BenchAgent(api_base="https://zjuapi.com/v1")
    result = await agent.evaluate("math", {"problem": "What is 2+2?", "solution": "4"})
    
    # result 很可能是 dict，不要用 .final_answer
    if isinstance(result, dict):
        print("Final answer:", result.get("final_answer"))
    else:
        print("Final answer:", result.final_answer)

        result.visualize_trajectory()

if __name__ == "__main__":
    asyncio.run(main())
    


