"""
语言模型包装器 - 提供统一的LLM接口

这个模块提供了对OpenAI API的包装，兼容原smolagents的接口。
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
import logging

logger = logging.getLogger(__name__)


class TokenMonitor:
    """Token使用监控器"""
    
    def __init__(self):
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.call_count = 0
    
    def update(self, input_tokens: int, output_tokens: int):
        """更新token统计"""
        self.total_input_token_count += input_tokens
        self.total_output_token_count += output_tokens
        self.call_count += 1
    
    def reset(self):
        """重置统计"""
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.call_count = 0


class OpenAIServerModel:
    """OpenAI模型包装器"""
    
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 300,
        max_completion_tokens: int = 4096,
        temperature: float = 0.2,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        初始化OpenAI模型
        
        Args:
            model_id: 模型ID
            api_key: API密钥
            api_base: API基础URL
            timeout: 超时时间
            max_completion_tokens: 最大生成token数
            temperature: 温度参数
            custom_role_conversions: 自定义角色转换映射
        """
        self.model_id = model_id
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.custom_role_conversions = custom_role_conversions or {}
        
        # 创建客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )
        
        # Token监控
        self.monitor = TokenMonitor()
        
        # 其他参数
        self.kwargs = kwargs
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息格式，应用自定义角色转换"""
        converted_messages = []
        
        for message in messages:
            converted_message = message.copy()
            
            # 应用角色转换
            if "role" in converted_message:
                original_role = converted_message["role"]
                if original_role in self.custom_role_conversions:
                    converted_message["role"] = self.custom_role_conversions[original_role]
            
            converted_messages.append(converted_message)
        
        return converted_messages
    
    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        **kwargs
    ) -> str:
        """同步调用模型"""
        try:
            # 转换消息格式
            converted_messages = self._convert_messages(messages)
            
            # 准备请求参数
            request_params = {
                "model": self.model_id,
                "messages": converted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_completion_tokens),
                **self.kwargs
            }
            
            # 添加停止序列
            if stop_sequences:
                request_params["stop"] = stop_sequences
            
            # 调用API
            response = self.client.chat.completions.create(**request_params)
            
            # 更新token统计
            if response.usage:
                self.monitor.update(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0
                )
            
            # 返回生成的文本
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    async def acall(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        **kwargs
    ) -> str:
        """异步调用模型"""
        try:
            # 转换消息格式
            converted_messages = self._convert_messages(messages)
            
            # 准备请求参数
            request_params = {
                "model": self.model_id,
                "messages": converted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_completion_tokens),
                **self.kwargs
            }
            
            # 添加停止序列
            if stop_sequences:
                request_params["stop"] = stop_sequences
            
            # 调用API
            response = await self.async_client.chat.completions.create(**request_params)
            
            # 更新token统计
            if response.usage:
                self.monitor.update(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0
                )
            
            # 返回生成的文本
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> ChatCompletion:
        """使用工具生成响应"""
        try:
            converted_messages = self._convert_messages(messages)
            
            request_params = {
                "model": self.model_id,
                "messages": converted_messages,
                "tools": tools,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_completion_tokens),
                **self.kwargs
            }
            
            response = self.client.chat.completions.create(**request_params)
            
            # 更新token统计
            if response.usage:
                self.monitor.update(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API with tools: {e}")
            raise
    
    async def agenerate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> ChatCompletion:
        """异步使用工具生成响应"""
        try:
            converted_messages = self._convert_messages(messages)
            
            request_params = {
                "model": self.model_id,
                "messages": converted_messages,
                "tools": tools,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_completion_tokens),
                **self.kwargs
            }
            
            response = await self.async_client.chat.completions.create(**request_params)
            
            # 更新token统计
            if response.usage:
                self.monitor.update(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API with tools: {e}")
            raise
    
    async def aclose(self):
        """关闭异步客户端"""
        if hasattr(self.async_client, 'aclose'):
            await self.async_client.aclose()
    
    def get_token_stats(self) -> Dict[str, int]:
        """获取token使用统计"""
        return {
            "total_input_tokens": self.monitor.total_input_token_count,
            "total_output_tokens": self.monitor.total_output_token_count,
            "total_tokens": self.monitor.total_input_token_count + self.monitor.total_output_token_count,
            "call_count": self.monitor.call_count
        }
