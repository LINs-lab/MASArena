import os
import asyncio
from openai import AsyncOpenAI
from process_pipeline import load_questions, extract_pattern_by_batch
import json
import yaml
from dotenv import load_dotenv

load_dotenv()

async def main():
    with open("preprocess_data/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    extract_pattern_system_prompt = prompts["extract_pattern_system_prompt"]
    extract_pattern_user_prompt = prompts["extract_pattern_user_prompt"]
    client = AsyncOpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_API_BASE"))
    gaia_validate_file = 'data/gaia_validate.jsonl'
    batch_size = 1
    need_deal_questions:set = {78}
    questions = {}
    with open(gaia_validate_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx not in need_deal_questions:
                continue
            if line.strip():
                data = json.loads(line)
                questions[f"question:{idx}"] = data.get("Question", "")
    
        """Extract pattern from a single question using GPT-4."""
    question_text = questions[f"question:{78}"]
    print(f"question_text:{question_text}")
    messages = [
        {"role": "system", "content": extract_pattern_system_prompt},
        {"role": "user", "content": extract_pattern_user_prompt.format(question=question_text)}
    ]
    
    response = await client.chat.completions.create(
        model="claude-3-7-sonnet-20250219-thinking",
        messages=messages,
        temperature=0.2,  
        max_tokens=4096
    )
    
    analysis_text = response.choices[0].message.content
    print(f"analysis_text:{analysis_text}")
    print(analysis_text)


if __name__ == "__main__":
    asyncio.run(main())