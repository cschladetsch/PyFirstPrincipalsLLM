#!/usr/bin/env python3
"""
Example usage of the Math LLM system.
This demonstrates various features and capabilities.
"""

import torch
from math_llm import MathLLM
from math_tokenizer import MathTokenizer
from expression_evaluator import ExpressionEvaluator
from value_storage import ValueStorage
from data_generator import MathDataGenerator

def basic_examples():
    """Demonstrate basic mathematical expression evaluation."""
    print("=== Basic Expression Evaluation ===")

    evaluator = ExpressionEvaluator()
    storage = ValueStorage("example_storage.db")
    storage.clear_all()

    expressions = [
        "a = 5",
        "b = 3",
        "c = a + b",
        "d = a * b",
        "e = sqrt(a*a + b*b)",
        "f = sin(0.5)",
        "g = log(10)",
        "h = abs(-7)"
    ]

    for expr in expressions:
        success, result, message = evaluator.parse_and_evaluate(expr)
        if success:
            print(f"✓ {message}")
            # Store in persistent storage
            var_name = expr.split('=')[0].strip()
            if isinstance(result, (int, float)):
                storage.set(var_name, result)
        else:
            print(f"✗ {message}")

    print(f"\nFinal variables: {evaluator.list_variables()}")
    print(f"Storage statistics: {storage.get_statistics()}")

def tokenizer_examples():
    """Demonstrate the mathematical tokenizer."""
    print("\n=== Tokenizer Examples ===")

    tokenizer = MathTokenizer()

    expressions = [
        "x = 3.14 * r^2",
        "y = sin(x) + cos(x)",
        "z = sqrt(a*a + b*b)",
        "result = (x + y) / z"
    ]

    for expr in expressions:
        print(f"\nExpression: {expr}")
        tokens = tokenizer.tokenize_expression(expr)
        print(f"Tokens: {tokens}")

        encoded = tokenizer.encode(expr, max_length=32)
        print(f"Encoded: {encoded.tolist()}")

        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")

def model_examples():
    """Demonstrate the transformer model capabilities."""
    print("\n=== Model Examples ===")

    from math_transformer import MathTransformer

    # Create a small model for demonstration
    model = MathTransformer(
        vocab_size=128,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64
    )

    tokenizer = MathTokenizer()

    # Example input
    input_text = "a = 5 ; b = a + 2"
    input_ids = tokenizer.encode(input_text, max_length=32).unsqueeze(0)
    attention_mask = tokenizer.create_attention_mask(input_ids[0]).unsqueeze(0)

    print(f"Input: {input_text}")
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        print(f"Output logits shape: {logits.shape}")

        # Generate next tokens
        generated = model.generate(input_ids, max_length=40, temperature=0.8)
        generated_text = tokenizer.decode(generated[0])
        print(f"Generated: {generated_text}")

def data_generation_examples():
    """Demonstrate training data generation."""
    print("\n=== Data Generation Examples ===")

    generator = MathDataGenerator()

    print("Simple expressions:")
    for i in range(5):
        expr = generator.generate_simple_expression()
        print(f"  {expr}")

    print("\nComplex expressions:")
    for i in range(5):
        expr = generator.generate_complex_expression()
        print(f"  {expr}")

    print("\nSequence example:")
    sequence = generator.generate_sequence(3)
    for i, expr in enumerate(sequence):
        print(f"  {i+1}. {expr}")

    print("\nTraining data sample:")
    data = generator.generate_training_data(3)
    for sample in data:
        print(f"  Input: '{sample['input']}'")
        print(f"  Target: '{sample['target']}'")
        print(f"  Full: '{sample['full_sequence']}'")
        print()

def advanced_examples():
    """Demonstrate advanced mathematical operations."""
    print("\n=== Advanced Examples ===")

    evaluator = ExpressionEvaluator()

    # Complex mathematical sequences
    expressions = [
        "pi = 3.14159",
        "radius = 5",
        "area = pi * radius^2",
        "circumference = 2 * pi * radius",
        "diagonal = sqrt(2) * radius",
        "volume_sphere = (4/3) * pi * radius^3"
    ]

    print("Calculating geometric properties:")
    for expr in expressions:
        success, result, message = evaluator.parse_and_evaluate(expr)
        if success:
            print(f"  {message}")
        else:
            print(f"  Error: {message}")

    # Trigonometric calculations
    print("\nTrigonometric calculations:")
    trig_expressions = [
        "angle = 0.5",
        "sin_val = sin(angle)",
        "cos_val = cos(angle)",
        "tan_val = tan(angle)",
        "identity_check = sin_val^2 + cos_val^2"
    ]

    for expr in trig_expressions:
        success, result, message = evaluator.parse_and_evaluate(expr)
        if success:
            print(f"  {message}")

def interactive_demo():
    """Demonstrate the interactive capabilities."""
    print("\n=== Interactive Demo ===")
    print("This would normally start an interactive session.")
    print("To try it, run: python math_llm.py --interactive")

    # Instead, simulate some interactions
    llm = MathLLM()

    test_expressions = [
        "x = 10",
        "y = x * 2",
        "z = sqrt(x^2 + y^2)"
    ]

    print("\nSimulated interactions:")
    for expr in test_expressions:
        result = llm.process_expression(expr)
        if result['success']:
            print(f">>> {expr}")
            print(f"✓ {result['message']}")
        else:
            print(f">>> {expr}")
            print(f"✗ {result['message']}")

def performance_test():
    """Test performance with batch operations."""
    print("\n=== Performance Test ===")

    evaluator = ExpressionEvaluator()
    generator = MathDataGenerator()

    # Generate a batch of expressions
    expressions = []
    for _ in range(100):
        expressions.append(generator.generate_simple_expression())

    print(f"Processing {len(expressions)} expressions...")

    import time
    start_time = time.time()

    results = evaluator.evaluate_batch(expressions[:10])  # Test with smaller batch

    end_time = time.time()

    successful = sum(1 for r in results if r['success'])
    print(f"Processed {len(results)} expressions in {end_time - start_time:.3f} seconds")
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")

def main():
    """Run all examples."""
    print("Math LLM - Example Usage")
    print("=" * 50)

    try:
        basic_examples()
        tokenizer_examples()
        model_examples()
        data_generation_examples()
        advanced_examples()
        interactive_demo()
        performance_test()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nTo get started:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train the model: python train.py")
        print("3. Use interactively: python math_llm.py --interactive")

    except Exception as e:
        print(f"\nError during examples: {e}")
        print("Make sure all dependencies are installed!")

if __name__ == "__main__":
    main()