"""
Utility functions for normalizing benchmark problem data.
"""
import os
from pathlib import Path
from typing import Dict, Any

def format_options(options: list) -> str:
    """
    Format options list into a readable string.
    
    Args:
        options: List of options
        
    Returns:
        Formatted options string
    """
    if not options:
        return ""
    
    formatted = []
    for i, option in enumerate(options):
        letter = chr(ord('A') + i)
        formatted.append(f"{letter}. {option}")
    
    return "\n".join(formatted)


def add_files_to_prompt(
    problem: Dict[str, Any], file_name: str = None
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, '..', '..', '..',  'data', 'files',"gaia","validate", file_name)
    file_path = Path(os.path.normpath(relative_path))
    if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
        problem["problem"] += f" Here are the necessary document files: {file_path}"

    elif file_path.suffix in [".jpg", ".jpeg", ".png"]:
        problem["problem"] += f" Here are the necessary image files: {file_path}"

    elif file_path.suffix in [".xlsx", "xls", ".csv"]:
        problem["problem"] += (
            f" Here are the necessary table files: {file_path}, for processing excel file,"
            " you can use the excel tool or write python code to process the file"
            " step-by-step and get the information."
        )
    elif file_path.suffix in [".py"]:
        problem["problem"] += f" Here are the necessary python files: {file_path}"

    else:
        problem["problem"] += f" Here are the necessary files: {file_path}"

    return problem["problem"]

def normalize_problem_keys(problem: Dict[str, Any], key_mapping: Dict[str, str], problem_index: int) -> Dict[str, Any]:
    """
    Normalizes the keys of a problem dictionary based on a provided key mapping.

    Args:
        problem: The original problem dictionary.
        key_mapping: A dictionary that maps standard keys ('id', 'problem', 'solution', etc.)
                     to the actual keys in the problem dictionary.
        problem_index: The index of the problem, used for generating a default ID if not present.

    Returns:
        A new dictionary with normalized keys.
    """
    normalized_problem = {}
    
    # Define a mapping from standard internal keys to the keys expected in the source data
    # and whether they are essential.
    key_definitions = {
        "id": "id",
        "problem": "problem",
        "context": "context",
        "solution": "solution",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": "test_imports",
        "instruction_id_list": "instruction_id_list",
        "kwargs": "kwargs"
    }

    for standard_key, source_key_name in key_definitions.items():
        source_key = key_mapping.get(source_key_name)
        if source_key and source_key in problem:
            normalized_problem[standard_key] = problem[source_key]

    # Handle options field for MMLU problems
    if "options" in problem:
        options_text = format_options(problem["options"])
        if "problem" in normalized_problem:
            normalized_problem["problem"] = f"{normalized_problem['problem']}\n\nOptions:\n{options_text}"
        else:
            normalized_problem["problem"] = f"Options:\n{options_text}"

    if "file_name" in problem and problem["file_name"]:
        file_name = problem["file_name"]
        normalized_problem["problem"] = add_files_to_prompt(normalized_problem, file_name)

    # Ensure a unique ID for the problem, generating one if not provided.
    if "id" not in normalized_problem:
        normalized_problem["id"] = f"problem_{problem_index + 1}"
        
    return normalized_problem 
