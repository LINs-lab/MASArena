"""
Utility functions for evaluators.
"""

from .sanitize import sanitize
from .answer_extraction import (
    extract_answer_generic,
    extract_answer_numeric, 
    extract_answer_simple_tags,
    extract_answer  # backward compatibility
)
from .metrics import (
    normalize_answer,
    calculate_f1_score,
    calculate_exact_match,
    calculate_multi_answer_f1
)

__all__ = [
    "sanitize",
    "extract_answer_generic",
    "extract_answer_numeric", 
    "extract_answer_simple_tags",
    "extract_answer",
    "normalize_answer",
    "calculate_f1_score", 
    "calculate_exact_match",
    "calculate_multi_answer_f1"
]
