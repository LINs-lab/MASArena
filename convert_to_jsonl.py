import json
import argparse

def convert_to_jsonl(input_file, output_file):
    """
    Converts a JSON file with a specific structure to a JSON-line file.

    Args:
        input_file (str): The path to the input JSON file.
        output_file (str): The path to the output JSON-line file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data.get('results', []):
            new_record = {
                "task_id": item.get("problem_id"),
                "model_answer": item.get("prediction"),
                "reasoning_trace": item.get("reasoning")
            }
            f_out.write(json.dumps(new_record) + '\n')
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GAIA results JSON to JSONL format.")
    parser.add_argument("input_file", help="The path to the input JSON file.")
    parser.add_argument("output_file", help="The path to the output JSONL file.")
    args = parser.parse_args()
    
    convert_to_jsonl(args.input_file, args.output_file)
