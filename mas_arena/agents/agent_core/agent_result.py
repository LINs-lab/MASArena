"""
AgentResult - 标准化的智能体执行结果数据结构

这个模块定义了智能体执行结果的标准化数据结构，用于统一不同智能体系统的返回格式。
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from mas_arena.visualization.mas_visualizer import MASVisualizer


@dataclass
class AgentResult:
    """
    标准化的智能体执行结果数据结构。
    
    该类提供了智能体执行结果的统一接口，支持字典式访问以保持向后兼容性。
    
    Attributes:
        final_answer (str): 智能体最终给出的答案。
        is_correct (bool): 是否正确（通过 scorer 或语义匹配判断）。
        trajectory (List[Dict[str, Any]]): 高层次、可读性强的执行轨迹，用于可视化。
        raw_responses (Dict[str, Any]): 原始响应数据，保留底层细节（如 steps、messages 等）。
        error (Optional[str]): 若执行失败，记录错误信息；否则为 None。
    
    Examples:
        >>> result = AgentResult(
        ...     final_answer="42",
        ...     is_correct=True,
        ...     trajectory=[{"agent": "manager", "action": "calculate"}],
        ...     raw_responses={"status": "success", "score": 1.0}
        ... )
        >>> result.final_answer
        '42'
        >>> result.get("score", 0)  # 字典式访问，从 raw_responses 中获取
        1.0
    """
    final_answer: str
    is_correct: bool
    trajectory: List[Dict[str, Any]]
    raw_responses: Dict[str, Any]
    error: Optional[str] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        字典式访问方法，用于向后兼容。
        
        首先尝试从 raw_responses 中获取，如果不存在则从对象属性中获取。
        
        Args:
            key: 要获取的键名
            default: 如果键不存在时返回的默认值
            
        Returns:
            对应的值，如果不存在则返回 default
        """
        # 优先从 raw_responses 中获取（保持向后兼容）
        if key in self.raw_responses:
            return self.raw_responses[key]
        # 然后尝试从对象属性中获取
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将 AgentResult 转换为字典格式。
        
        合并对象属性和 raw_responses 的内容，raw_responses 中的值会覆盖同名属性。
        
        Returns:
            包含所有数据的字典
        """
        result_dict = asdict(self)
        # 合并 raw_responses 的内容（raw_responses 中的值优先级更高）
        result_dict.update(self.raw_responses)
        return result_dict
    
    def visualize_trajectory(self, output_file: Optional[str] = None, open_browser: bool = True) -> Any:
        """
        可视化执行轨迹。
        
        Args:
            output_file: 输出文件路径，如果为 None 则使用默认路径
            open_browser: 是否在浏览器中打开可视化结果
            
        Returns:
            可视化结果，如果失败则返回 None
        """
        visualization_file = self.raw_responses.get("visualization_file")
        if visualization_file is None:
            print("Not found visualizations file")
            return None
        return MASVisualizer().visualize(visualization_file, output_file=output_file, open_browser=open_browser)

