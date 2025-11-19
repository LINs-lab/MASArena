from typing import Dict, Any
from smolagents import Tool


class FinalAnswerTool(Tool):
    """Final answer submission tool."""

    name = "final_answer"
    description = "Use this tool to provide the final answer to the user's question."

    inputs = {
        "answer": {
            "type": "string",
            "description": "The final answer to provide to the user.",
        }
    }
    output_type = "string"

    def forward(self, answer: str) -> str:
        """Return the final answer."""
        return f"Final answer: {answer}"


def test_final_answer_tool_basic():
    """测试基本功能"""
    tool = FinalAnswerTool()

    result = tool.forward("The capital of France is Paris.")
    expected = "Final answer: The capital of France is Paris."

    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✅ test_final_answer_tool_basic passed")


def test_final_answer_tool_empty_string():
    """测试空字符串输入"""
    tool = FinalAnswerTool()

    result = tool.forward("")
    expected = "Final answer: "

    assert result == expected, f"Expected '{expected}', got '{result}'"
    print("✅ test_final_answer_tool_empty_string passed")


def test_final_answer_tool_special_characters():
    """测试特殊字符和长文本"""
    tool = FinalAnswerTool()

    answer = "42! @#$%^&*() 🌍\n\t✓ 数学答案：$E=mc^2$"
    result = tool.forward(answer)
    expected = f"Final answer: {answer}"

    assert result == expected, f"Expected special char output to match"
    print("✅ test_final_answer_tool_special_characters passed")


def test_tool_metadata():
    """测试工具的元数据是否符合规范"""
    tool = FinalAnswerTool()

    # 检查类属性
    assert hasattr(tool, "name")
    assert tool.name == "final_answer"

    assert hasattr(tool, "description")
    assert isinstance(tool.description, str) and len(tool.description) > 0

    assert hasattr(tool, "inputs")
    assert isinstance(tool.inputs, dict)
    assert "answer" in tool.inputs
    assert tool.inputs["answer"]["type"] == "string"

    assert hasattr(tool, "output_type")
    assert tool.output_type == "string"

    print("✅ test_tool_metadata passed")


def test_integration_like_usage():
    """模拟在 agent 系统中的典型使用方式"""
    tool = FinalAnswerTool()

    user_question = "What is 2 + 2?"
    computed_answer = "4"

    # Agent 决定调用 final_answer 工具
    response = tool.forward(computed_answer)

    assert response.startswith("Final answer:")
    assert computed_answer in response
    print("✅ test_integration_like_usage passed")


if __name__ == "__main__":
    print("🧪 Running tests for FinalAnswerTool...\n")

    try:
        test_final_answer_tool_basic()
        test_final_answer_tool_empty_string()
        test_final_answer_tool_special_characters()
        test_tool_metadata()
        test_integration_like_usage()

        print("\n🎉 All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
