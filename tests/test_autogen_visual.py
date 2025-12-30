# test.py
import sys
import os
import asyncio
import logging
from datetime import datetime
os.environ["CHROMA_TELEMETRY"] = "false"  # 关闭 Chroma 遥测

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
async def main():
    from mas_arena.agents.autogen_newcore import AutoGen
    # from mas_arena.agents.smolagent import SmolagentsAgent
    agent = AutoGen()
    problem_data = {
        "id": "test_001",
        "problem": "What is 2+2?", 
        "solution": "4"
    }
    print(f"--- 正在开始评估问题: {problem_data['id']} ---")
    try:
        result = await agent.evaluate(
            problem=problem_data,
            evaluator_name = "math",
        )
        print("\n" + "="*50)
        print("评估完成！")
        
        # 习惯上先检查 raw_responses 获取详细信息
        raw = result.raw_responses
        print(f"状态: {raw.get('status')}")
        print(f"是否正确: {result.is_correct}")
        print(f"提取的答案: {result.final_answer}")
        print(f"消耗时间: {raw.get('execution_time_ms', 0):.2f} ms")
        
        # 打印 Token 消耗 (LLM Usage)
        usage = raw.get("llm_usage", {})
        print(f"Token 消耗: {usage}")

        # 5. 可视化轨迹 (如果 AgentResult 支持)
        result.visualize_trajectory()
    except Exception as e:
        print(f"运行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        # # result 很可能是 dict，不要用 .final_answer
        # if isinstance(result, dict):
        #     print("Final answer:", result.get("final_answer"))
        # else:
        #     print("Final answer:", result.final_answer)
        #     # print("visualizations_url:", result.visualizations_url)
        #     result.visualize_trajectory()

if __name__ == "__main__":
    asyncio.run(main())
    
# import asyncio
# import time
# import json
# import sys
# import os
# from typing import Dict, Any

# from mas_arena.agents.autogen_newcore import AutoGen

# async def main():
#     # 1. 构造 problem 参数 (必须包含 id 和 problem 文本)
#     test_problem = {
#         "id": "task_001",
#         "problem": "请问 2+2 等于多少？",
#         "solution": "4" # meta_memory 可能会用到
#     }

#     # 2. 构造可选的 kwargs (例如 metrics_registry)
#     extra_args = {
#         "metrics_registry": None, # 或者传入你的 registry 对象
#         "custom_setting": "value"
#     }
#     agent = AutoGen() 
#     try:
#         # 3. 调用异步函数
#         # 注意：由于是在类外部，通常是通过实例调用： result = await agent_system.evaluate(...)
#         print(f"开始评估问题: {test_problem['id']}...")
        
#         result = await agent.evaluate(
#             problem=test_problem, 
#             **extra_args
#         )

#         # 4. 处理返回的 AgentResult 对象
#         print("--- 运行结果 ---")
#         print(f"状态: {'成功' if result.is_correct else '失败'}")
#         print(f"提取的答案: {result.final_answer}")
#         if result.error:
#             print(f"错误信息: {result.error}")
        
#         # 打印轨迹 (Trajectory)
#         print(f"执行轨迹步数: {len(result.trajectory)}")

#     except Exception as e:
#         print(f"调用过程中发生崩溃: {e}")

# # 运行异步主函数
# if __name__ == "__main__":
#     asyncio.run(main())