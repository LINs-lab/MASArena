import asyncio
from typing import List, Dict, Any
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

async def to_agent_loop_output(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, Any]],
    response_length: int
) -> AgentLoopOutput:
    """
    Converts a list of messages into the AgentLoopOutput format required by VERL.
    This is a simplified version for demonstration.
    """
    loop = asyncio.get_running_loop()

    # The prompt is all messages except the last one
    prompt_messages = messages[:-1]
    # The response is the last message
    response_messages = messages[-1:]
    
    # Let the tokenizer handle the full conversation templating
    full_ids = await loop.run_in_executor(
        None,
        lambda: tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        ),
    )
    
    prompt_ids = await loop.run_in_executor(
        None,
        lambda: tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True
        ),
    )
    
    # The response IDs are the part of the full sequence after the prompt
    response_ids = full_ids[len(prompt_ids):]
    
    # For PPO, the response mask is 1 for agent-generated tokens
    response_mask = [1] * len(response_ids)
    
    max_len = min(response_length, len(response_ids))
    
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:max_len],
        response_mask=response_mask[:max_len],
        num_turns=len(messages) // 2,
        metrics={},
    )