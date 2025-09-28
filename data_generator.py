import random
import numpy as np
from typing import List, Tuple, Dict
from expression_evaluator import ExpressionEvaluator

class MathDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.evaluator = ExpressionEvaluator()
        self.variables = ['a', 'b', 'c', 'x', 'y', 'z', 'm', 'n', 'p', 'q']

    def generate_number(self, max_value: int = 100, allow_float: bool = True) -> str:
        if allow_float and random.random() < 0.3:
            return f"{random.uniform(-max_value, max_value):.2f}"
        else:
            return str(random.randint(-max_value, max_value))

    def generate_simple_expression(self) -> str:
        templates = [
            "{var} = {num1} + {num2}",
            "{var} = {num1} - {num2}",
            "{var} = {num1} * {num2}",
            "{var} = {num1} / {num2}",
            "{var} = {num1}",
            "{var} = {var2} + {num}",
            "{var} = {var2} - {num}",
            "{var} = {var2} * {num}",
            "{var} = {var2} / {num}",
            "{var} = {var2} + {var3}",
            "{var} = {var2} - {var3}",
            "{var} = {var2} * {var3}",
        ]

        template = random.choice(templates)
        var = random.choice(self.variables)
        var2 = random.choice([v for v in self.variables if v != var])
        var3 = random.choice([v for v in self.variables if v not in [var, var2]])

        expression = template.format(
            var=var,
            var2=var2,
            var3=var3,
            num=self.generate_number(20),
            num1=self.generate_number(20),
            num2=self.generate_number(20, allow_float=False)
        )

        return expression

    def generate_complex_expression(self) -> str:
        templates = [
            "{var} = ({num1} + {num2}) * {num3}",
            "{var} = {num1} * {num2} + {num3}",
            "{var} = ({var2} + {num1}) * {num2}",
            "{var} = {var2} * {num1} - {var3} * {num2}",
            "{var} = ({var2} - {var3}) / {num}",
            "{var} = sqrt({num})",
            "{var} = sin({num})",
            "{var} = cos({num})",
            "{var} = exp({num})",
            "{var} = log({num})",
            "{var} = abs({var2} - {num})",
            "{var} = max({var2}, {num})",
            "{var} = min({var2}, {num})",
            "{var} = pow({num1}, {num2})",
        ]

        template = random.choice(templates)
        var = random.choice(self.variables)
        var2 = random.choice([v for v in self.variables if v != var])
        var3 = random.choice([v for v in self.variables if v not in [var, var2]])

        expression = template.format(
            var=var,
            var2=var2,
            var3=var3,
            num=self.generate_number(10, allow_float=False),
            num1=self.generate_number(10, allow_float=False),
            num2=self.generate_number(5, allow_float=False),
            num3=self.generate_number(10, allow_float=False)
        )

        return expression

    def generate_sequence(self, length: int = 5) -> List[str]:
        sequence = []
        for _ in range(length):
            if random.random() < 0.7:
                sequence.append(self.generate_simple_expression())
            else:
                sequence.append(self.generate_complex_expression())
        return sequence

    def generate_training_data(self, num_samples: int = 1000) -> List[Dict[str, any]]:
        data = []
        for _ in range(num_samples):
            seq_length = random.randint(1, 10)
            expressions = self.generate_sequence(seq_length)

            input_expressions = expressions[:-1] if len(expressions) > 1 else []
            target_expression = expressions[-1] if expressions else ""

            if input_expressions:
                input_text = " ; ".join(input_expressions)
            else:
                input_text = ""

            data.append({
                'input': input_text,
                'target': target_expression,
                'full_sequence': " ; ".join(expressions)
            })

        return data

    def generate_evaluation_samples(self) -> List[Dict[str, str]]:
        samples = [
            {'input': '', 'target': 'a = 5'},
            {'input': 'a = 5', 'target': 'b = a + 3'},
            {'input': 'a = 5 ; b = a + 3', 'target': 'c = a * b'},
            {'input': 'x = 10', 'target': 'y = x / 2'},
            {'input': 'x = 10 ; y = x / 2', 'target': 'z = x - y'},
            {'input': 'm = 7', 'target': 'n = m * m'},
            {'input': 'p = 3 ; q = 4', 'target': 'r = sqrt(p * p + q * q)'},
            {'input': 'a = 2 ; b = 3', 'target': 'c = pow(a, b)'},
            {'input': 'x = 45', 'target': 'y = sin(x)'},
            {'input': 'a = -5', 'target': 'b = abs(a)'},
        ]
        return samples

    def create_batch(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 64
    ) -> Dict[str, any]:
        input_ids = []
        target_ids = []
        attention_masks = []

        for sample in data:
            input_text = sample['input']
            target_text = sample['target']

            if input_text:
                input_encoded = tokenizer.encode(input_text, max_length=max_length)
            else:
                input_encoded = tokenizer.encode("", max_length=max_length)

            target_encoded = tokenizer.encode(target_text, max_length=max_length)

            attention_mask = tokenizer.create_attention_mask(input_encoded)

            input_ids.append(input_encoded)
            target_ids.append(target_encoded)
            attention_masks.append(attention_mask)

        import torch
        return {
            'input_ids': torch.stack(input_ids),
            'target_ids': torch.stack(target_ids),
            'attention_mask': torch.stack(attention_masks)
        }