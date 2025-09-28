import unittest
import torch
from math_tokenizer import MathTokenizer
from expression_evaluator import ExpressionEvaluator
from value_storage import ValueStorage
from math_transformer import MathTransformer
from data_generator import MathDataGenerator

class TestMathTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MathTokenizer()

    def test_basic_tokenization(self):
        expr = "a = 1 + 2"
        tokens = self.tokenizer.tokenize_expression(expr)
        expected = ['a', '=', '1', '+', '2']
        self.assertEqual(tokens, expected)

    def test_encoding_decoding(self):
        expr = "x = 5 * 3"
        encoded = self.tokenizer.encode(expr)
        decoded = self.tokenizer.decode(encoded)
        self.assertIn("x", decoded)
        self.assertIn("5", decoded)
        self.assertIn("3", decoded)

    def test_complex_expression(self):
        expr = "result = sqrt(a*a + b*b)"
        tokens = self.tokenizer.tokenize_expression(expr)
        self.assertIn('sqrt', tokens)
        self.assertIn('result', tokens)

class TestExpressionEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ExpressionEvaluator()

    def test_simple_assignment(self):
        success, result, message = self.evaluator.parse_and_evaluate("a = 5")
        self.assertTrue(success)
        self.assertEqual(result, 5)
        self.assertEqual(self.evaluator.get_variable("a"), 5)

    def test_arithmetic_operations(self):
        self.evaluator.parse_and_evaluate("a = 10")
        success, result, message = self.evaluator.parse_and_evaluate("b = a + 5")
        self.assertTrue(success)
        self.assertEqual(result, 15)

    def test_mathematical_functions(self):
        success, result, message = self.evaluator.parse_and_evaluate("x = sqrt(16)")
        self.assertTrue(success)
        self.assertEqual(result, 4.0)

    def test_complex_expression(self):
        self.evaluator.parse_and_evaluate("a = 3")
        self.evaluator.parse_and_evaluate("b = 4")
        success, result, message = self.evaluator.parse_and_evaluate("c = sqrt(a*a + b*b)")
        self.assertTrue(success)
        self.assertEqual(result, 5.0)

    def test_invalid_expression(self):
        success, result, message = self.evaluator.parse_and_evaluate("invalid = 1 +")
        self.assertFalse(success)

class TestValueStorage(unittest.TestCase):
    def setUp(self):
        self.storage = ValueStorage("test_storage.db")
        self.storage.clear_all()

    def tearDown(self):
        self.storage.clear_all()

    def test_set_and_get(self):
        self.storage.set("test_var", 42)
        value = self.storage.get("test_var")
        self.assertEqual(value, 42)

    def test_update(self):
        self.storage.set("test_var", 10)
        self.storage.update("test_var", 20)
        value = self.storage.get("test_var")
        self.assertEqual(value, 20)

    def test_delete(self):
        self.storage.set("test_var", 123)
        self.assertTrue(self.storage.exists("test_var"))
        self.storage.delete("test_var")
        self.assertFalse(self.storage.exists("test_var"))

    def test_statistics(self):
        self.storage.set("var1", 1)
        self.storage.set("var2", 2.5)
        self.storage.set("var3", "test")
        stats = self.storage.get_statistics()
        self.assertEqual(stats['total_variables'], 3)

class TestMathTransformer(unittest.TestCase):
    def setUp(self):
        self.model = MathTransformer(
            vocab_size=128,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            max_position_embeddings=64
        )
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def test_forward_pass(self):
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 128, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = self.model(input_ids, attention_mask)
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (batch_size, seq_len, 128))

    def test_generation(self):
        input_ids = torch.tensor([[2, 5, 25]])  # <BOS> = 1
        generated = self.model.generate(input_ids, max_length=20, temperature=1.0)
        self.assertGreater(generated.shape[1], input_ids.shape[1])

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = MathDataGenerator()

    def test_simple_expression_generation(self):
        expr = self.generator.generate_simple_expression()
        self.assertIn("=", expr)

    def test_complex_expression_generation(self):
        expr = self.generator.generate_complex_expression()
        self.assertIn("=", expr)

    def test_sequence_generation(self):
        sequence = self.generator.generate_sequence(5)
        self.assertEqual(len(sequence), 5)
        for expr in sequence:
            self.assertIn("=", expr)

    def test_training_data_generation(self):
        data = self.generator.generate_training_data(10)
        self.assertEqual(len(data), 10)
        for sample in data:
            self.assertIn('input', sample)
            self.assertIn('target', sample)
            self.assertIn('full_sequence', sample)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MathTokenizer()
        self.evaluator = ExpressionEvaluator()
        self.storage = ValueStorage("integration_test.db")
        self.storage.clear_all()

    def tearDown(self):
        self.storage.clear_all()

    def test_full_pipeline(self):
        expression = "x = 10"

        tokens = self.tokenizer.tokenize_expression(expression)
        self.assertIn('x', tokens)
        self.assertIn('=', tokens)
        self.assertIn('1', tokens)
        self.assertIn('0', tokens)

        encoded = self.tokenizer.encode(expression)
        self.assertIsInstance(encoded, torch.Tensor)

        decoded = self.tokenizer.decode(encoded)
        self.assertIn("x", decoded)

        success, result, message = self.evaluator.parse_and_evaluate(expression)
        self.assertTrue(success)
        self.assertEqual(result, 10)

        self.storage.set("x", result)
        stored_value = self.storage.get("x")
        self.assertEqual(stored_value, 10)

    def test_variable_dependency(self):
        expressions = [
            "a = 5",
            "b = a + 3",
            "c = a * b"
        ]

        for expr in expressions:
            success, result, message = self.evaluator.parse_and_evaluate(expr)
            if not success:
                print(f"Failed expression: {expr}, message: {message}")
            self.assertTrue(success, f"Expression '{expr}' failed: {message}")

        self.assertEqual(self.evaluator.get_variable("a"), 5)
        self.assertEqual(self.evaluator.get_variable("b"), 8)
        self.assertEqual(self.evaluator.get_variable("c"), 40)

if __name__ == '__main__':
    print("Running Math LLM Tests...")
    unittest.main(verbosity=2)