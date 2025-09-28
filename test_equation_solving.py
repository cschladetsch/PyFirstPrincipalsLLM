#!/usr/bin/env python3
"""
Test script for equation solving and unknown variable functionality
"""

from expression_evaluator import ExpressionEvaluator

def test_basic_equations():
    """Test basic equation solving"""
    evaluator = ExpressionEvaluator()

    print("Testing Basic Equation Solving")
    print("=" * 40)

    # Test simple linear equation
    success, result, message = evaluator.parse_and_evaluate("2*x + 5 = 11")
    print(f"2*x + 5 = 11 -> {message}")

    # Test quadratic equation
    success, result, message = evaluator.parse_and_evaluate("x^2 - 5*x + 6 = 0")
    print(f"x^2 - 5*x + 6 = 0 -> {message}")

    # Test with known variables
    evaluator.set_variable("a", 2)
    evaluator.set_variable("b", 3)
    success, result, message = evaluator.parse_and_evaluate("a*x + b = 9")
    print(f"a*x + b = 9 (where a=2, b=3) -> {message}")

    print()

def test_parentheses():
    """Test parentheses handling"""
    evaluator = ExpressionEvaluator()

    print("Testing Parentheses")
    print("=" * 40)

    # Test basic parentheses
    success, result, message = evaluator.parse_and_evaluate("result = 2 * (3 + 4)")
    print(f"result = 2 * (3 + 4) -> {message}")

    # Test nested parentheses
    success, result, message = evaluator.parse_and_evaluate("complex = (2 + 3) * (4 + 5)")
    print(f"complex = (2 + 3) * (4 + 5) -> {message}")

    # Test with functions
    success, result, message = evaluator.parse_and_evaluate("trig = sin(2 * (pi/4))")
    print(f"trig = sin(2 * (pi/4)) -> {message}")

    print()

def test_unknowns():
    """Test unknown variable handling"""
    evaluator = ExpressionEvaluator()

    print("Testing Unknown Variables")
    print("=" * 40)

    # Test expression with unknown
    success, result, message = evaluator.parse_and_evaluate("expr1 = 2*x + 3*y")
    print(f"expr1 = 2*x + 3*y -> {message}")

    # Now assign values to unknowns
    success, result, message = evaluator.parse_and_evaluate("x = 5")
    print(f"x = 5 -> {message}")

    success, result, message = evaluator.parse_and_evaluate("y = 2")
    print(f"y = 2 -> {message}")

    # Evaluate expression with now-known variables
    success, result, message = evaluator.parse_and_evaluate("final = 2*x + 3*y")
    print(f"final = 2*x + 3*y -> {message}")

    print()

def test_solve_for_variable():
    """Test solving for specific variables"""
    evaluator = ExpressionEvaluator()

    print("Testing Solve for Variable")
    print("=" * 40)

    # Test linear equation
    success, result, message = evaluator.solve_for_variable("3*x + 7 = 22", "x")
    print(f"Solve 3*x + 7 = 22 for x -> {message}")

    # Test quadratic equation
    success, result, message = evaluator.solve_for_variable("x^2 - 4 = 0", "x")
    print(f"Solve x^2 - 4 = 0 for x -> {message}")

    # Test with known variables
    evaluator.set_variable("a", 2)
    evaluator.set_variable("c", 8)
    success, result, message = evaluator.solve_for_variable("a*x^2 = c", "x")
    print(f"Solve a*x^2 = c for x (a=2, c=8) -> {message}")

    print()

def test_complex_expressions():
    """Test complex mathematical expressions"""
    evaluator = ExpressionEvaluator()

    print("Testing Complex Expressions")
    print("=" * 40)

    # Test with trigonometric functions
    success, result, message = evaluator.parse_and_evaluate("pi = 3.14159")
    print(f"pi = 3.14159 -> {message}")

    success, result, message = evaluator.parse_and_evaluate("angle = pi/4")
    print(f"angle = pi/4 -> {message}")

    success, result, message = evaluator.parse_and_evaluate("sine_val = sin(angle)")
    print(f"sine_val = sin(angle) -> {message}")

    # Test with square root
    success, result, message = evaluator.parse_and_evaluate("hypotenuse = sqrt(3^2 + 4^2)")
    print(f"hypotenuse = sqrt(3^2 + 4^2) -> {message}")

    print()

def main():
    """Run all tests"""
    print("Math LLM Enhanced Expression Testing")
    print("=" * 50)
    print()

    test_basic_equations()
    test_parentheses()
    test_unknowns()
    test_solve_for_variable()
    test_complex_expressions()

    print("All tests completed!")

if __name__ == "__main__":
    main()