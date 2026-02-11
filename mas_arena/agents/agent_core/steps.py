"""
执行步骤类 - 用于追踪代理的执行过程

这些类用于记录代理在执行任务时的各个步骤，包括动作、计划和任务。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import json


logger = logging.getLogger(__name__)


@dataclass
class BaseStep:
    """基础步骤类"""
    step_number: int = 0
    timestamp: float = field(default_factory=time.time)
    agent_name: str = "unknown"
    error: Optional[str] = None
    
    def dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "error": self.error,
            "step_type": self.__class__.__name__,
        }


@dataclass
class ActionStep(BaseStep):
    """动作步骤 - 记录工具调用和结果"""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    model_output: Optional[str] = None
    model_output_message: Optional[Dict[str, Any]] = None
    
    def dict(self) -> Dict[str, Any]:
        base_dict = super().dict()
        base_dict.update({
            "tool_calls": self.tool_calls,
            "observations": self.observations,
            "model_output": self.model_output,
        })
        return base_dict


@dataclass
class PlanningStep(BaseStep):
    """计划步骤 - 记录代理的计划过程"""
    plan: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    model_output: Optional[str] = None
    model_output_message_facts: Optional[Dict[str, Any]] = None
    model_output_message_plan: Optional[Dict[str, Any]] = None
    
    def dict(self) -> Dict[str, Any]:
        base_dict = super().dict()
        base_dict.update({
            "plan": self.plan,
            "facts": self.facts,
            "model_output": self.model_output,
        })
        return base_dict


@dataclass
class TaskStep(BaseStep):
    """任务步骤 - 记录委托给其他代理的任务"""
    task: str = ""
    assigned_agent: str = ""
    result: Optional[str] = None
    model_output: Optional[str] = None
    
    def dict(self) -> Dict[str, Any]:
        base_dict = super().dict()
        base_dict.update({
            "task": self.task,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "model_output": self.model_output,
        })
        return base_dict


class StepMemory:
    """步骤内存管理器"""
    
    def __init__(self):
        self.steps: List[BaseStep] = []
        self._step_counter = 0
    
    def add_step(self, step: BaseStep) -> None:
        """添加新步骤，并把完整步骤信息记录到日志"""
        self._step_counter += 1
        step.step_number = self._step_counter
        self.steps.append(step)

        # 记录结构化 step 日志，方便离线分析
        try:
            payload = json.dumps(step.dict(), ensure_ascii=False)
        except Exception:
            try:
                payload = str(step.dict())
            except Exception:
                payload = f"<unserializable step type={type(step).__name__}>"

        if len(payload) > 4000:
            payload = payload[:4000] + "...<truncated>"

        logger.info("STEP_LOG %s", payload)
    
    def get_steps(self, step_type: Optional[type] = None) -> List[BaseStep]:
        """获取指定类型的步骤"""
        if step_type is None:
            return self.steps.copy()
        return [step for step in self.steps if isinstance(step, step_type)]
    
    def clear(self) -> None:
        """清空步骤历史"""
        self.steps.clear()
        self._step_counter = 0
    
    def get_last_step(self, step_type: Optional[type] = None) -> Optional[BaseStep]:
        """获取最后一个步骤"""
        steps = self.get_steps(step_type)
        return steps[-1] if steps else None
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """转换为字典格式"""
        return [step.dict() for step in self.steps]
