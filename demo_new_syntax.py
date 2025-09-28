#!/usr/bin/env python3
"""
Demo script showing the new x? syntax for solving equations
"""

import subprocess
import sys

def run_command(cmd):
    """Run a math_llm.py command and return the output"""
    try:
        result = subprocess.run(
            [sys.executable, "math_llm.py"] + cmd.split()[1:],  # Skip "python"
            capture_output=True,
            text=True,
            cwd="."
        )
        return result.stdout.strip().split('\n')[-1]  # Get the last line
    except Exception as e:
        return f"Error: {e}"

def main():
    print("Math LLM - New x? Syntax Demo")
    print("=" * 40)
    print()

    demos = [
        ("Basic variable assignment:", "a = 5"),
        ("Expression with parentheses:", "result = 2*(3+4)"),
        ("Expression with unknowns:", "expr = 2*x + 3*y"),
        ("Linear equation solving:", "x? 2*x + 5 = 11"),
        ("Quadratic equation:", "y? y^2 - 9 = 0"),
        ("Cubic equation:", "z? z^3 - 27 = 0"),
        ("Complex equation:", "t? t^2 + 2*t + 5 = 0"),
    ]

    for description, command in demos:
        print(f"{description}")
        print(f"  Input:  {command}")
        output = run_command(f"python {command}")
        print(f"  Output: {output}")
        print()

    print("Interactive Usage Examples:")
    print("  > a = 10")
    print("  > b = 5")
    print("  > a?              # Shows: a = 10")
    print("  > x?              # Shows: x is undefined")
    print("  > x? a*x + b = 25 # Solves: x = 1.5")
    print("  > x?              # Shows: x = 1.5")
    print("  > y? y^2 - a*y + b = 0")
    print()

    print("The new syntax is much more intuitive:")
    print("  Old: /solve x 'x^2 + 2*x + 1 = 0'")
    print("  New: x? x^2 + 2*x + 1 = 0")
    print("  Query: x? (shows current value or 'undefined')")

if __name__ == "__main__":
    main()