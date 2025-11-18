from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
from typing import Tuple


def calculate_score(expected_output: str, prediction: str) -> Tuple[int, str]:
    """
    Calculate score using Math-Verify's verification capabilities.
    For math problems, gold answers can be various formats, but predictions may contain full explanations.
    """
    try:
        # Parse the gold answer with both LaTeX and expression extractors
        gold_parsed = parse(
            expected_output,
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
            ],
            extraction_mode="any_match",
            fallback_mode="first_match",
        )

        # Parse the model's answer with recommended configuration for evaluation
        answer_parsed = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        basic_latex=True,  # Enable basic LaTeX command replacements
                        units=True,  # Remove units (helpful for mixed unit answers)
                        malformed_operators=False,  # Don't fix malformed operators for strict evaluation
                        nits=False,  # Don't apply small formatting fixes for strict evaluation
                        boxed="all",  # Extract from all boxed environments
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=True,  # Allow extraction without LaTeX anchors
                ),
                ExprExtractionConfig(),  # Also try plain expression extraction
            ],
            extraction_mode="any_match",
            fallback_mode="first_match",
        )

        if len(gold_parsed) != 0 and len(answer_parsed) != 0:
            try:
                # Verify the answers match
                is_correct = verify(gold_parsed, answer_parsed)
                # Extract the actual value from the parsed result
                extracted_value = str(answer_parsed[0]) if isinstance(answer_parsed, list) else str(answer_parsed)
                return int(is_correct), extracted_value
            except Exception as e:
                print(f"Verification failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
    except Exception as e:
        print(f"Parsing failed: {e}")

    # Enhanced fallback: try to extract numbers from text
    import re

    # Look for numbers that might be answers
    number_patterns = [
        r"(?:final answer|answer|result|solution)[\s:]*(\d+)",  # "final answer is 81"
        r"(\d+)\.?\s*$",  # number at end of text
        r"\b(\d+)\b",  # any standalone number
    ]

    for pattern in number_patterns:
        matches = re.findall(pattern, prediction, re.IGNORECASE)
        if matches:
            candidate = matches[-1]  # Take the last match
            try:
                if int(candidate) == int(expected_output):
                    return 1, candidate
            except (ValueError, TypeError):
                continue

    # Final fallback: simple string comparison
    expected = str(expected_output).strip()
    pred = str(prediction).strip()
    if expected == pred:
        return 1, pred
    return 0, pred


# Test examples for calculate_score method
if __name__ == "__main__":
    # Comprehensive test cases for various mathematical expressions
    test_cases = [
        # Basic arithmetic
        {"expected": "42", "prediction": "The answer is $\\boxed{42}$", "description": "Simple integer in boxed LaTeX"},
        # Fractions - ensure LaTeX environment
        {
            "expected": "$\\frac{1}{2}$",  # Gold answer in LaTeX environment
            "prediction": "After simplification, we get $\\boxed{\\frac{1}{2}}$",
            "description": "Fraction in boxed LaTeX",
        },
        # Decimal vs fraction equivalence
        {
            "expected": "0.5",
            "prediction": "The result is $\\frac{1}{2}$",
            "description": "Decimal vs fraction equivalence",
        },
        # Square roots - ensure LaTeX environment
        {
            "expected": "$\\sqrt{2}$",  # Gold answer in LaTeX environment
            "prediction": "The answer is $\\boxed{\\sqrt{2}}$",
            "description": "Square root expression",
        },
        # Complex square roots
        {
            "expected": "$2\\sqrt{3}$",  # Gold answer in LaTeX environment
            "prediction": "Simplifying gives us $\\boxed{2\\sqrt{3}}$",
            "description": "Coefficient with square root",
        },
        # Polynomials
        {
            "expected": "$x^2 + 2x + 1$",  # Gold answer in LaTeX environment
            "prediction": "Expanding $(x+1)^2$ gives $\\boxed{x^2 + 2x + 1}$",
            "description": "Polynomial expression",
        },
        # Equivalent polynomial forms
        {
            "expected": "$(x+1)^2$",  # Gold answer in LaTeX environment
            "prediction": "The factored form is $\\boxed{x^2 + 2x + 1}$",
            "description": "Equivalent polynomial forms",
        },
        # Trigonometric functions
        {
            "expected": "$\\sin(\\frac{\\pi}{2})$",  # Gold answer in LaTeX environment
            "prediction": "The value is $\\boxed{\\sin(\\frac{\\pi}{2})}$",
            "description": "Trigonometric function",
        },
        # Logarithms
        {
            "expected": "$\\log_2(8)$",  # Gold answer in LaTeX environment
            "prediction": "Since $2^3 = 8$, we have $\\boxed{\\log_2(8)} = 3$",
            "description": "Logarithmic expression",
        },
        # Set notation
        {
            "expected": "$\\{1, 2, 3\\}$",  # Gold answer in LaTeX environment
            "prediction": "The solution set is $\\boxed{\\{1, 2, 3\\}}$",
            "description": "Set notation",
        },
        # Interval notation
        {
            "expected": "$(0, 1)$",  # Gold answer in LaTeX environment
            "prediction": "The interval is $\\boxed{(0, 1)}$",
            "description": "Open interval notation",
        },
        # Matrix
        {
            "expected": "$\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}$",  # Gold answer in LaTeX environment
            "prediction": "The identity matrix is $\\boxed{\\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}}$",
            "description": "2x2 identity matrix",
        },
        # Complex numbers
        {
            "expected": "$3 + 4i$",  # Gold answer in LaTeX environment
            "prediction": "The complex number is $\\boxed{3 + 4i}$",
            "description": "Complex number",
        },
        # Equations
        {
            "expected": "$x = 5$",  # Gold answer in LaTeX environment
            "prediction": "Solving the equation gives $\\boxed{x = 5}$",
            "description": "Simple equation",
        },
        # Plain text number
        {"expected": "42", "prediction": "The final answer is 42.", "description": "Plain text number"},
        # Number with explanation
        {
            "expected": "15",
            "prediction": "After calculating 3 × 5, we get 15 as the final result.",
            "description": "Number with explanation",
        },
        # Percentage
        {
            "expected": "$25\\%$",
            "prediction": "The percentage is $\\boxed{25\\%}$",
            "description": "Percentage notation",
        },
        # Scientific notation
        {
            "expected": "$1.5 \\times 10^3$",
            "prediction": "In scientific notation: $\\boxed{1.5 \\times 10^3}$",
            "description": "Scientific notation",
        },
        # Wrong answers for testing
        {
            "expected": "$\\frac{1}{2}$",
            "prediction": "The answer is $\\boxed{\\frac{1}{3}}$",
            "description": "Wrong fraction (should fail)",
        },
        {"expected": "42", "prediction": "The answer is 43", "description": "Wrong number (should fail)"},
        # Edge cases
        {"expected": "0", "prediction": "The result is zero: $\\boxed{0}$", "description": "Zero value"},
        {"expected": "-5", "prediction": "The negative answer is $\\boxed{-5}$", "description": "Negative number"},
        # Multiple expressions in text
        {
            "expected": "7",
            "prediction": "We first calculate 2 + 3 = 5, then 5 + 2 = $\\boxed{7}$",
            "description": "Multiple numbers, boxed answer correct",
        },
        {
            "expected": "$-\\frac{17}{19}$",
            "prediction": " so $\\sin \\theta = \\boxed{-\\frac{17}{19}}.$",
            "description": "Final answer in boxed LaTeX",
        },
    ]

    print("Testing Math-Verify calculate_score function:")
    print("=" * 70)

    correct_count = 0
    total_count = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        expected = test_case["expected"]
        prediction = test_case["prediction"]
        description = test_case["description"]

        score, extracted = calculate_score(expected, prediction)

        if score == 1:
            correct_count += 1

        status = "✓" if score == 1 else "✗"
        print(f"Test {i:2d}: {description}")
        print(f"Expected:   {expected}")
        print(f"Prediction: {prediction}")
        print(f"Extracted:  {extracted}")
        print(f"Score:      {score} ({status})")
        print("-" * 70)

    print(f"\nSummary: {correct_count}/{total_count} tests passed ({correct_count / total_count * 100:.1f}%)")

    # Expected failures (for verification)
    expected_failures = ["Wrong fraction (should fail)", "Wrong number (should fail)"]
    actual_failures = [
        test_cases[i]["description"]
        for i, test_case in enumerate(test_cases)
        if calculate_score(test_case["expected"], test_case["prediction"])[0] == 0
    ]

    print(f"\nExpected failures: {len(expected_failures)}")
    print(f"Actual failures: {len(actual_failures)}")
    print("Failed tests:", actual_failures)
