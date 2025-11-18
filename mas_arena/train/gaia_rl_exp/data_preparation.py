import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

def load_gaia_jsonl(file_path):
    """Loads a GAIA .jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_rl_format(data):
    """
    Converts GAIA data to a format suitable for RL training.
    We need a 'prompt' and the necessary fields for our reward function.
    """
    rl_dataset = {
        "prompt": [],
        "ground_truth": [],
        "file_name": [],
    }
    for item in tqdm(data):
        # The 'prompt' is the full conversation history before the agent's turn.
        # For GAIA, this is usually just the question.
        prompt = [
            {"role": "user", "content": item["Question"]}
        ]
        rl_dataset["prompt"].append(prompt)
        rl_dataset["ground_truth"].append(item["Final answer"])
        rl_dataset["file_name"].append(item.get("file_name", "")) # For context
        
    return pd.DataFrame(rl_dataset)

def main():
    parser = argparse.ArgumentParser(description="Convert GAIA dataset to Parquet for RL training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the GAIA .jsonl file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the Parquet file.")
    parser.add_argument("--split", type=str, default="train", help="The name of the split (e.g., train, test).")
    
    args = parser.parse_args()

    print(f"Loading data from: {args.data_path}")
    raw_data = load_gaia_jsonl(args.data_path)
    
    print("Converting to RL format...")
    df = convert_to_rl_format(raw_data)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.split}.parquet")
    
    print(f"Saving to Parquet file: {output_path}")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    main()