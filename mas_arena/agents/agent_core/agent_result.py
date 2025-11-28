# final_answer: str: 最终答案。
# is_correct: bool: 答案是否正确。
# trajectory: List[Dict]: 清晰的执行轨迹，供 visualize_trajectory 使用。
# raw_responses: Dict: 包含底层的、详细的智能体日志（如 manager_agent_steps），与现有系统衔接。
# error: Optional[str]: 如果执行出错，记录错误信息。
# 分析函数 (Analysis Functions): 这些是独立的辅助函数，接收 AgentResult 对象进行处理。
# visualize_trajectory(result: AgentResult): 将 result.trajectory 可视化，清晰地打印出来。
# analyze_error(result: AgentResult): 对失败的 result 进行分析和归因。


from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from mas_arena.visualization.mas_visualizer import MASVisualizer
@dataclass
class AgentResult:
    """
    标准化的智能体执行结果数据结构。
    
    Attributes:
        final_answer (str): 智能体最终给出的答案。
        is_correct (bool): 是否正确（通过 scorer 或语义匹配判断）。
        trajectory (List[Dict]): 高层次、可读性强的执行轨迹，用于可视化。
        raw_responses (Dict[str, Any]): 原始响应数据，保留底层细节（如 steps、messages 等）。
        error (Optional[str]): 若执行失败，记录错误信息；否则为 None。
    """
    final_answer: str
    is_correct: bool
    trajectory: List[Dict[str, Any]]
    raw_responses: Dict[str, Any]
    error: Optional[str] = None
     
    def visualize_trajectory(self, output_file=None, open_browser=True):
        if self.raw_responses["visualization_file"] is None:  
            print("Not found visualizations file")
            return 
        else:
            return MASVisualizer().visualize(self.raw_responses["visualization_file"], output_file=output_file, open_browser=open_browser)

