"""
Evaluation metrics utilities.

This module provides common metric calculations used across different evaluators.
"""

import re
import string
from collections import Counter
from typing import List, Any


def normalize_answer(s: Any) -> str:
    """
    Normalize answer text for evaluation.
    Standard normalization: lowercase -> remove articles/punctuation -> collapse whitespace.
    Used by DROP, HotpotQA and similar benchmarks.
    
    Args:
        s: The text to normalize
        
    Returns:
        Normalized text string
    """
    s = str(s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def calculate_f1_score(gold: str, pred: str, normalize_fn=None) -> float:
    """
    Calculate token-level F1 score between gold and predicted answers.
    
    Args:
        gold: Ground truth answer
        pred: Predicted answer
        normalize_fn: Optional normalization function. If None, uses normalize_answer.
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    if normalize_fn is None:
        normalize_fn = normalize_answer
        
    gold_toks: List[str] = normalize_fn(gold).split()
    pred_toks: List[str] = normalize_fn(pred).split()

    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(gold: str, pred: str, normalize_fn=None) -> float:
    """
    Calculate exact match score between gold and predicted answers.
    
    Args:
        gold: Ground truth answer
        pred: Predicted answer  
        normalize_fn: Optional normalization function. If None, uses normalize_answer.
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if normalize_fn is None:
        normalize_fn = normalize_answer
        
    return 1.0 if normalize_fn(gold) == normalize_fn(pred) else 0.0


def calculate_multi_answer_f1(gold_answers: List[str], pred_answers: List[str], normalize_fn=None) -> float:
    """
    Calculate the best F1 score when there are multiple possible gold and predicted answers.
    Used by DROP and similar benchmarks that support multiple valid answers.
    
    Args:
        gold_answers: List of valid ground truth answers
        pred_answers: List of predicted answers
        normalize_fn: Optional normalization function. If None, uses normalize_answer.
        
    Returns:
        Best F1 score found between any gold-pred pair
    """
    if not gold_answers or not pred_answers:
        return 0.0
        
    scores = [
        calculate_f1_score(gold, pred, normalize_fn)
        for gold in gold_answers for pred in pred_answers
    ]
    
    return max(scores) if scores else 0.0
