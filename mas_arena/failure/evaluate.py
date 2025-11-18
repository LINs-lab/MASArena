import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional


def read_predictions(evaluation_file: str) -> Dict[str, Dict[str, str]]:
    """
    Read predictions from the evaluation file.
    
    Args:
        evaluation_file: Path to the evaluation file containing predictions
        
    Returns:
        Dictionary mapping file names to their predictions
    """
    predictions = {}
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by file separators
    file_sections = re.split(r'=== File: (.+?) ===', content)
    
    for i in range(1, len(file_sections), 2):
        if i + 1 < len(file_sections):
            filename = file_sections[i].strip()
            file_content = file_sections[i + 1].strip()
            
            # Extract error agent and step from the content
            error_agent_match = re.search(r'Error Agent: (.+)', file_content)
            error_step_match = re.search(r'Error Step: (.+)', file_content)
            
            error_agent = error_agent_match.group(1).strip() if error_agent_match else "Unknown"
            error_step = error_step_match.group(1).strip() if error_step_match else "Unknown"
            
            predictions[filename] = {
                'error_agent': error_agent,
                'error_step': error_step
            }
    
    return predictions


def read_actual_data(data_path: str) -> Dict[str, Dict[str, str]]:
    """
    Read actual error data from annotated JSON files.
    
    Args:
        data_path: Path to the directory containing annotated JSON files
        
    Returns:
        Dictionary mapping file names to their actual error data
    """
    actual_data = {}
    
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} does not exist.")
        return actual_data
    
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            filepath = os.path.join(data_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract actual error information from the JSON structure
                # This assumes the JSON has fields like 'actual_error_agent' and 'actual_error_step'
                # Adjust based on your actual data structure
                actual_error_agent = data.get('actual_error_agent', 'Unknown')
                actual_error_step = data.get('actual_error_step', 'Unknown')
                
                actual_data[filename] = {
                    'error_agent': str(actual_error_agent),
                    'error_step': str(actual_error_step)
                }
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {filename}: {e}")
                continue
    
    return actual_data


def evaluate_accuracy(predictions: Dict[str, Dict[str, str]], 
                     actual_data: Dict[str, Dict[str, str]]) -> Tuple[float, float]:
    """
    Evaluate the accuracy of predictions against actual data.
    
    Args:
        predictions: Dictionary of predicted error data
        actual_data: Dictionary of actual error data
        
    Returns:
        Tuple of (agent_accuracy, step_accuracy)
    """
    if not predictions or not actual_data:
        print("Warning: No predictions or actual data available for evaluation.")
        return 0.0, 0.0
    
    agent_correct = 0
    step_correct = 0
    total_files = 0
    
    # Find common files between predictions and actual data
    common_files = set(predictions.keys()) & set(actual_data.keys())
    
    if not common_files:
        print("Warning: No common files found between predictions and actual data.")
        return 0.0, 0.0
    
    print(f"Evaluating {len(common_files)} common files...")
    
    for filename in common_files:
        pred = predictions[filename]
        actual = actual_data[filename]
        
        total_files += 1
        
        # Check agent accuracy
        if pred['error_agent'].lower() == actual['error_agent'].lower():
            agent_correct += 1
        
        # Check step accuracy
        if pred['error_step'].lower() == actual['error_step'].lower():
            step_correct += 1
        
        print(f"File: {filename}")
        print(f"  Predicted Agent: {pred['error_agent']} | Actual Agent: {actual['error_agent']} | {'✓' if pred['error_agent'].lower() == actual['error_agent'].lower() else '✗'}")
        print(f"  Predicted Step: {pred['error_step']} | Actual Step: {actual['error_step']} | {'✓' if pred['error_step'].lower() == actual['error_step'].lower() else '✗'}")
        print()
    
    agent_accuracy = agent_correct / total_files if total_files > 0 else 0.0
    step_accuracy = step_correct / total_files if total_files > 0 else 0.0
    
    return agent_accuracy, step_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate failure attribution accuracy.")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing actual/annotated data JSON files."
    )
    parser.add_argument(
        "--evaluation_file",
        type=str,
        required=True,
        help="Path to the evaluation file containing predictions."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file {args.evaluation_file} does not exist.")
        return
    
    print(f"Reading predictions from: {args.evaluation_file}")
    predictions = read_predictions(args.evaluation_file)
    print(f"Found predictions for {len(predictions)} files.")
    
    print(f"Reading actual data from: {args.data_path}")
    actual_data = read_actual_data(args.data_path)
    print(f"Found actual data for {len(actual_data)} files.")
    
    agent_accuracy, step_accuracy = evaluate_accuracy(predictions, actual_data)
    
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Agent Accuracy: {agent_accuracy:.2%} ({agent_accuracy:.4f})")
    print(f"Step Accuracy: {step_accuracy:.2%} ({step_accuracy:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()