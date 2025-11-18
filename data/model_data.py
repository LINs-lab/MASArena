"""
Model parameter and activation data for memory estimation.

This module contains synthetic data about model sizes and their typical
activated parameters during inference, including parameter format and size information.
"""

MODEL_DATA = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "parameter_size_b": 8.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "parameter_size_b": 70.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "meta-llama/Llama-3.1-405B": {
        "parameter_size_b": 405.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 16384,
        "num_attention_heads": 128,
        "num_hidden_layers": 126,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "parameter_size_b": 3.0,
        "activated_size_b": 0,  
        "bytes_per_parameter": 2,
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "parameter_size_b": 1.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "parameter_size_b": 7.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "Qwen/Qwen-7B": {
        "parameter_size_b": 7.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "dtype": "float16"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "parameter_size_b": 7.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "dtype": "float16"
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "parameter_size_b": 3.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_hidden_layers": 36,
        "num_key_value_heads": 2,
        "dtype": "float16"
    },
    "Qwen/Qwen2.5-14B": {  
        "parameter_size_b": 14.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
        "Qwen/Qwen2.5-32B": {
        "parameter_size_b": 32.5,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },   
    "Qwen/Qwen3-4B": {
        "parameter_size_b": 4.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "Qwen/Qwen3-8B": {
        "parameter_size_b": 8.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "Qwen/Qwen3-14B": {
        "parameter_size_b": 14.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "Qwen/Qwen3-32B": {
        "parameter_size_b": 32.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "Qwen/QwQ-32B-Preview": {
        "parameter_size_b": 32.5,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },    
    "deepseek-ai/DeepSeek-R1": {
        "parameter_size_b": 671,
        "activated_size_b": 37,  # Utilizes 1 shared expert and 256 routed experts, with 37B parameters activated per token
        "bytes_per_parameter": 2,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "dtype": "float16"
    },
    "deepseek-ai/DeepSeek-V3": {
        "parameter_size_b": 671,
        "activated_size_b": 37,  # Utilizes 1 shared expert and 256 routed experts, with 37B parameters activated per token
        "bytes_per_parameter": 2,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "dtype": "float16"
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "parameter_size_b": 32.5,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "dtype": "float16"
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "parameter_size_b": 14.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "dtype": "float16" 
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "parameter_size_b": 7.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "dtype": "float16"
    }, 
    "lmsys/longchat-7b-16k": {
        "parameter_size_b": 7.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "dtype": "float16"
    },  
    "Sao10K/L3-8B-Lunaris-v1": {
        "parameter_size_b": 8.0,
        "activated_size_b": 0, 
        "bytes_per_parameter": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "dtype": "float16"
    }
} 