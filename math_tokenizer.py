import re
from typing import List, Dict, Tuple, Optional
import torch

class MathTokenizer:
    def __init__(self):
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
        }

        self.operators = {
            '=': 5,
            '+': 6,
            '-': 7,
            '*': 8,
            '/': 9,
            '^': 10,
            '(': 11,
            ')': 12,
            ',': 13,
            '.': 14,
        }

        self.functions = {
            'sin': 15,
            'cos': 16,
            'tan': 17,
            'log': 18,
            'exp': 19,
            'sqrt': 20,
            'abs': 21,
            'min': 22,
            'max': 23,
            'pow': 24,
        }

        self.digits = {str(i): 25 + i for i in range(10)}

        self.variables_start_id = 35
        self.variables = {}
        self.next_variable_id = self.variables_start_id

        self.token_to_id = {
            **self.special_tokens,
            **self.operators,
            **self.functions,
            **self.digits,
        }

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.vocab_size = 128

    def tokenize_expression(self, expression: str) -> List[str]:
        expression = expression.replace(' ', '').lower()

        pattern = r'(\d+\.?\d*|[a-z_]\w*|[+\-*/^()=,.])'
        tokens = re.findall(pattern, expression)

        processed_tokens = []
        for token in tokens:
            if re.match(r'^\d+\.?\d*$', token):
                if '.' in token:
                    parts = token.split('.')
                    for digit in parts[0]:
                        processed_tokens.append(digit)
                    processed_tokens.append('.')
                    for digit in parts[1]:
                        processed_tokens.append(digit)
                else:
                    for digit in token:
                        processed_tokens.append(digit)
            elif token in self.functions:
                processed_tokens.append(token)
            elif token in self.operators:
                processed_tokens.append(token)
            elif re.match(r'^[a-z_]\w*$', token):
                processed_tokens.append(token)
            else:
                processed_tokens.append(token)

        return processed_tokens

    def encode(self, expression: str, max_length: Optional[int] = None) -> torch.Tensor:
        tokens = self.tokenize_expression(expression)

        ids = [self.special_tokens['<BOS>']]

        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            elif re.match(r'^[a-z_]\w*$', token):
                if token not in self.variables:
                    if self.next_variable_id < self.vocab_size:
                        self.variables[token] = self.next_variable_id
                        self.id_to_token[self.next_variable_id] = token
                        self.next_variable_id += 1
                    else:
                        ids.append(self.special_tokens['<UNK>'])
                        continue
                ids.append(self.variables[token])
            else:
                ids.append(self.special_tokens['<UNK>'])

        ids.append(self.special_tokens['<EOS>'])

        if max_length:
            if len(ids) < max_length:
                ids.extend([self.special_tokens['<PAD>']] * (max_length - len(ids)))
            else:
                ids = ids[:max_length-1] + [self.special_tokens['<EOS>']]

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        if ids.dim() > 1:
            ids = ids.squeeze()

        tokens = []
        for id_val in ids.tolist():
            if id_val in self.id_to_token:
                token = self.id_to_token[id_val]
                if token not in ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<UNK>']:
                    tokens.append(token)

        result = ''
        i = 0
        while i < len(tokens):
            if tokens[i] in self.digits:
                number = ''
                while i < len(tokens) and (tokens[i] in self.digits or tokens[i] == '.'):
                    number += tokens[i]
                    i += 1
                result += number
                i -= 1
            else:
                result += tokens[i]

            if i < len(tokens) - 1:
                if tokens[i] in self.functions:
                    pass
                elif tokens[i+1] not in ['+', '-', '*', '/', '^', ')', ',', '='] and tokens[i] not in ['(', '=']:
                    if not (tokens[i] in self.digits and tokens[i+1] in self.digits):
                        if not (tokens[i] == '.' or tokens[i+1] == '.'):
                            result += ' '
            i += 1

        return result

    def create_attention_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids != self.special_tokens['<PAD>']).float()

    def get_variable_names(self) -> List[str]:
        return list(self.variables.keys())

    def reset_variables(self):
        self.variables = {}
        self.next_variable_id = self.variables_start_id
        for key in list(self.id_to_token.keys()):
            if key >= self.variables_start_id:
                del self.id_to_token[key]