"""
Answer extraction utilities for evaluators.

This module provides common functions for extracting answers from model outputs
across different benchmark evaluators.
"""

import re


def extract_answer_generic(text: str) -> str:
    """
    Generic answer extraction with comprehensive fallback patterns.
    Suitable for most text-based benchmarks like HotpotQA, BBH, etc.

    Args:
        text: The model's output text

    Returns:
        The extracted answer (e.g., "(A)", "True", text content)
    """
    text = text.strip()

    # Primary pattern: Content within <answer>...</answer> tags
    tag_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
    match = re.search(tag_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: "Final Answer: <answer>"
    final_answer_pattern = r"Final Answer:\s*(.+)"
    match = re.search(final_answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: Look for multiple-choice options (e.g., (A), A, [A])
    option_pattern = r"\([A-Z]\)|[A-Z]\b|\[[A-Z]\]"
    matches = re.findall(option_pattern, text, re.DOTALL)
    if matches:
        last_match = matches[-1]
        # Normalize to (A) format
        if not last_match.startswith("("):
            last_match = f"({last_match[-1]})"
        return last_match.strip()

    # Fallback: Look for boolean values
    boolean_pattern = r"\b(True|False)\b"
    boolean_matches = re.findall(boolean_pattern, text, re.DOTALL)
    if boolean_matches:
        return boolean_matches[-1].strip()

    # Fallback: Look for sequence completions (e.g., "> ) }", "] ] ]")
    sequence_pattern = r"([>\]\}\)\[]+\s*)+"
    sequence_matches = re.findall(sequence_pattern, text, re.DOTALL)
    if sequence_matches:
        return sequence_matches[-1].strip()

    # Fallback: Last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return lines[-1]

    # Final fallback: Return stripped text
    return text.strip()


def extract_answer_numeric(text: str) -> str:
    """
    Extract numeric answers, suitable for math problems like AIME.
    
    Args:
        text: The model's output text
        
    Returns:
        The extracted numeric answer
    """
    # Try to extract the last number (int/float)
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
    if matches:
        return matches[-1].replace(",", "").strip()
    
    # Fallback: last non-empty line
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def extract_answer_simple_tags(text: str) -> str:
    """
    Simple answer extraction for evaluators that primarily use <answer> tags.
    Suitable for MMLU-Pro and similar benchmarks.
    
    Args:
        text: The model's output text
        
    Returns:
        The extracted answer
    """
    # Try to extract answer from <answer> tags, allowing for whitespace
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no tags found, return original response
    return text.strip()


# Backward compatibility aliases
extract_answer = extract_answer_generic
