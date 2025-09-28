import sympy as sp
import re
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

class ExpressionEvaluator:
    def __init__(self):
        self.variables: Dict[str, Union[float, int, sp.Symbol]] = {}
        self.history: list = []
        self.functions = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'log': sp.log,
            'exp': sp.exp,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'min': sp.Min,
            'max': sp.Max,
            'pow': lambda x, y: x**y,
        }

    def parse_and_evaluate(self, expression: str) -> Tuple[bool, Any, Optional[str]]:
        try:
            expression = expression.strip()
            self.history.append(expression)

            if '=' in expression:
                parts = expression.split('=', 1)
                if len(parts) != 2:
                    return False, None, "Invalid assignment format"

                left_side = parts[0].strip()
                right_side = parts[1].strip()

                if not re.match(r'^[a-zA-Z_]\w*$', left_side):
                    return False, None, f"Invalid variable name: {left_side}"

                result = self._evaluate_expression(right_side)
                self.variables[left_side] = result

                return True, result, f"{left_side} = {result}"

            else:
                result = self._evaluate_expression(expression)
                return True, result, str(result)

        except Exception as e:
            return False, None, f"Error: {str(e)}"

    def _evaluate_expression(self, expr: str) -> Union[float, int]:
        expr = self._preprocess_expression(expr)

        local_vars = {}
        for var_name, var_value in self.variables.items():
            if isinstance(var_value, (int, float)):
                local_vars[var_name] = var_value
            else:
                local_vars[var_name] = sp.Symbol(var_name)

        sympy_expr = sp.sympify(expr, locals={**local_vars, **self.functions})

        if hasattr(sympy_expr, 'free_symbols'):
            free_symbols = sympy_expr.free_symbols
        else:
            free_symbols = set()
        undefined_vars = [str(s) for s in free_symbols if str(s) not in self.variables]

        if undefined_vars:
            for var in undefined_vars:
                self.variables[var] = sp.Symbol(var)
                local_vars[var] = sp.Symbol(var)

            return sympy_expr
        else:
            # Substitute all known variable values
            substitutions = {}
            for var_name, var_value in self.variables.items():
                if isinstance(var_value, (int, float)):
                    substitutions[sp.Symbol(var_name)] = var_value

            if substitutions and hasattr(sympy_expr, 'subs'):
                sympy_expr = sympy_expr.subs(substitutions)

            if hasattr(sympy_expr, 'evalf'):
                result = sympy_expr.evalf()
                return float(result)
            else:
                return float(sympy_expr)

    def _preprocess_expression(self, expr: str) -> str:
        expr = expr.replace('^', '**')

        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)
        expr = re.sub(r'\)(\w)', r')*\1', expr)
        expr = re.sub(r'(\w)\(', r'\1*(', expr)

        for func in self.functions:
            expr = re.sub(f'{func}\\*\\(', f'{func}(', expr)

        return expr

    def get_variable(self, name: str) -> Optional[Union[float, int, sp.Symbol]]:
        return self.variables.get(name)

    def set_variable(self, name: str, value: Union[float, int]):
        if not re.match(r'^[a-zA-Z_]\w*$', name):
            raise ValueError(f"Invalid variable name: {name}")
        self.variables[name] = value

    def clear_variables(self):
        self.variables = {}
        self.history = []

    def list_variables(self) -> Dict[str, Any]:
        result = {}
        for name, value in self.variables.items():
            if isinstance(value, (int, float)):
                result[name] = value
            else:
                result[name] = str(value)
        return result

    def get_history(self) -> list:
        return self.history.copy()

    def evaluate_batch(self, expressions: list) -> list:
        results = []
        for expr in expressions:
            success, value, message = self.parse_and_evaluate(expr)
            results.append({
                'expression': expr,
                'success': success,
                'value': value,
                'message': message
            })
        return results