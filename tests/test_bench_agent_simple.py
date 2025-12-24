import asyncio
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from mas_arena.agents.bench_agent import BenchAgent

async def debug_agent():
    
    print("=== 1. 检查组件初始化 ===")
    agent = BenchAgent(
        model="gpt-4.1",
        manager_tools=["python_interpreter", "final_answer"],
        search_tools=["search"],
        verbosity_level=3
    )
    
    # 验证工具是否正确注入
    print(f"管理工具列表: {[t.name for t in agent.manager_tools]}")
    print(f"搜索工具列表: {[t.name for t in agent.search_tools]}")
    
    print("\n=== 3. 测试单步运行 (run_agent_step) ===")
    # 绕过复杂的 evaluate 逻辑，直接测试单步能力
    # step_problem = "计算 25 * 4 等于多少？"
    step_problem = "计算 12345 乘以 98765 的精确结果，请务必使用 python 工具计算。"
    
    try:
        # 模拟 run_agent 内部调用的核心逻辑
        print("启动单步任务...")
        step_result = await agent.run_agent_step(step_problem, additional_args={})
        
        print(f"单步运行最终回答: {step_result.get('final_answer')}")
        print(f"管理代理步数: {len(step_result.get('manager_agent_steps', []))}")
        
        # 打印具体的步骤内容，看看 Agent 到底在想什么
        for i, step in enumerate(step_result.get('manager_agent_steps', [])):
            print(f"步骤 {i+1} 内容: {step.get('thought', '无思考内容')}")
            print(f"步骤 {i+1} 工具调用: {step.get('tool_calls', '无工具调用')}")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        await agent.aclose()

if __name__ == "__main__":
    asyncio.run(debug_agent())