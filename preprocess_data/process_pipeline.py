import asyncio
import os
import random
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json
from typing import Dict, List,Optional
from tqdm.asyncio import tqdm_asyncio
import yaml
import re

with open("preprocess_data/prompts.yaml", "r") as f:
    prompts = yaml.safe_load(f)
    extract_pattern_system_prompt = prompts["extract_pattern_system_prompt"]
    extract_pattern_user_prompt = prompts["extract_pattern_user_prompt"]
    extract_meta_pattern_system_prompt = prompts["extract_meta_pattern_system_prompt"]
    extract_meta_pattern_user_prompt = prompts["extract_meta_pattern_user_prompt"]

def _parse_analysis_to_dict(analysis_text: str) -> Dict[str, str]:
    """
    Parses the pattern_analysis text into a dictionary.
    This version supports both single-line and multi-line key-value formats.
    """
    analysis_dict = {}

    # Define all possible key variations
    keys = [
        r"\*{0,2}Question Type\*{0,2}",
        r"\*{0,2}Key Elements\*{0,2}",
        r"\*{0,2}Context Information\*{0,2}",
        r"\*{0,2}Pattern\*{0,2}"
    ]
    keys_pattern = "|".join(keys)

    # Regex pattern to match key-value pairs, allowing multiline values
    pattern = re.compile(
        rf"(?P<key>{keys_pattern})\s*[:=]?\s*\n*(?P<value>.*?)(?=\n\s*(?:{keys_pattern})\s*[:=]?|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern.finditer(analysis_text):
        key = match.group('key').strip().replace('*', '')
        value = match.group('value').strip()

        # Normalize keys
        if "Question Type" in key:
            analysis_dict['question_type'] = value
        elif "Key Elements" in key:
            analysis_dict['key_elements'] = value
        elif "Context Information" in key:
            analysis_dict['context_information'] = value
        elif "Pattern" in key:
            # Remove surrounding quotes if any
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            analysis_dict['pattern'] = value

    return analysis_dict

async def load_questions(file_path: str) -> Dict[str, str]:
    """Load questions from a JSONL file."""
    questions = {}
    #need_ready_questions:set = {10,11,18,19,21,22,47,68,75,78,80,81,83,84,87,90,97,98,118,122,144,149}
    need_ready_questions:set ={1}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            # if idx not in need_ready_questions:
            #     continue
            if line.strip():
                data = json.loads(line)
                questions[f"question:{idx}"] = data.get("Question", "")
    
    # random.seed(42)
    # questions = {k: v for k, v in random.sample(list(questions.items()), k=1)}
    return questions

async def extract_pattern(client: AsyncOpenAI, question_id: str, question_text: str) -> Dict:
    """Extract pattern from a single question using GPT-4."""
    try:
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
        analysis_dict = _parse_analysis_to_dict(analysis_text)
        
        return {
            "question_id": question_id,
            "question": question_text,
            "pattern_analysis": analysis_dict if analysis_dict else analysis_text,
            "raw_analysis_text": analysis_text 
        }
    except Exception as e:
        return {
            "question_id": question_id,
            "question": question_text,
            "error": f"Error processing question: {str(e)}"
        }

async def extract_pattern_by_batch(client: AsyncOpenAI, questions: Dict[str, str], batch_size: int = 5) -> List[Dict]:
    """Process questions in batches to avoid rate limiting."""
    all_patterns = []
    question_items = list(questions.items())
    
    for i in range(0, len(question_items), batch_size):
        batch = question_items[i:i + batch_size]
        tasks = [
            extract_pattern(client, q_id, q_text) 
            for q_id, q_text in batch
        ]
        batch_results = await tqdm_asyncio.gather(*tasks)
        all_patterns.extend(batch_results)
    
    return all_patterns

async def get_meta_pattern(client: AsyncOpenAI, pattern_data: Dict[str, str], prompts: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Generates a meta-pattern (μπ) and returns a structured dictionary."""

    try:
        user_content = prompts['extract_meta_pattern_user_prompt'].format(
            question_type=pattern_data["question_type"],
            Pattern=pattern_data["pattern"]
        )
        
        messages = [
            {"role": "system", "content": prompts['extract_meta_pattern_system_prompt']},
            {"role": "user", "content": user_content}
        ]
        
        response = await client.chat.completions.create(
            model="claude-3-7-sonnet-20250219-thinking",
            messages=messages,
            temperature=0.1,
            max_tokens=4096
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Strict cleaning of the response
        match = re.search(r'Meta-Pattern\s*=\s*"(.*)"', raw_response)
        if match:
            meta_pattern = match.group(1)
        else:
            # Fallback: aggressively remove prefixes and quotes
            meta_pattern = re.sub(r'^(Meta-Pattern\s*=\s*"?|Meta-Pattern\s*:\s*"?|\*\*Output Meta-Pattern:\*\*\s*"?)+', '', raw_response)
            meta_pattern = meta_pattern.strip().strip('"')

        return {
            "question_id": pattern_data["question_id"],
            "question_text": pattern_data["question_text"],
            "question_type": pattern_data["question_type"],
            "meta_pattern": meta_pattern,
            "original_pattern": pattern_data["pattern"]
        }

    except Exception as e:
        print(f"Error during API call for question:{pattern_data['id']}: {e}")
        return {
            "question_id": pattern_data["question_id"],
            "question_text": pattern_data["question_text"],
            "question_type": pattern_data["question_type"],
            "meta_pattern": f"Error generating meta-pattern: {e}",
            "original_pattern": pattern_data.get("pattern", "N/A")
        }

async def extract_meta_pattern_by_batch(client: AsyncOpenAI, patterns: List[Dict[str, str]], batch_size: int, output_file: str):
    """Processes patterns in batches and writes results to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm_asyncio(range(0, len(patterns), batch_size), desc="Processing Batches"):
            batch = patterns[i:i+batch_size]
            tasks = [get_meta_pattern(client, p_data, prompts) for p_data in batch]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for res in results:
                    # 处理异常情况
                    if isinstance(res, Exception):
                        print(f"Warning: Task failed with error: {res}")
                        continue
                    elif res:
                        f.write(json.dumps(res) + '\n')
                        
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

async def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    api_base = os.getenv("ANTHROPIC_API_BASE")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    gaia_validate_file = 'data/gaia_validate.jsonl'
    pattern_file = 'preprocess_data/question_pattern_0909.jsonl'
    meta_pattern_file = 'preprocess_data/meta_patterns_0909.jsonl'
    batch_size = 5
    
    questions = await load_questions(gaia_validate_file)
    question_type_patterns = await extract_pattern_by_batch(client, questions, batch_size)

    with open(pattern_file, 'w', encoding='utf-8') as f:
        for item in question_type_patterns:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    patterns = []

    patterns = [{**(v.get("pattern_analysis") or {}), "question_id": v.get("question_id"), "question_text": v.get("question")}
    for v in question_type_patterns]
    

    await extract_meta_pattern_by_batch(client, patterns, batch_size, meta_pattern_file)

    print(f"\nProcessing complete. Results saved to {meta_pattern_file}")


if __name__ == "__main__":
    asyncio.run(main())