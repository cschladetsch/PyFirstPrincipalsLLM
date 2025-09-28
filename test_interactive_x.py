#!/usr/bin/env python3
"""
Test the x? syntax in a programmatic way to simulate interactive session
"""

from math_llm import MathLLM

def test_x_query():
    """Test the x? syntax for querying variable values"""
    print("Testing x? syntax for variable queries")
    print("=" * 40)

    llm = MathLLM()

    # Test undefined variable
    print("Testing undefined variable:")
    result = llm.process_expression("x")
    print(f"x -> {result['message']}")

    # Define some variables
    print("\nDefining variables:")
    result = llm.process_expression("a = 5")
    print(f"a = 5 -> {result['message']}")

    result = llm.process_expression("b = 10")
    print(f"b = 10 -> {result['message']}")

    result = llm.process_expression("c = a + b")
    print(f"c = a + b -> {result['message']}")

    # Now test the interactive functionality with a quick demo
    print(f"\nIn interactive mode, you can now use:")
    print(f"  x? -> Shows value of x or 'x is undefined'")
    print(f"  a? -> Shows value of a")
    print(f"  x? 2*x + 5 = 11 -> Solves equation for x")

    print(f"\nCurrent variables in session:")
    vars = llm.evaluator.list_variables()
    for name, value in vars.items():
        print(f"  {name} = {value}")

if __name__ == "__main__":
    test_x_query()