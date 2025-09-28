import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import sys

from math_transformer import MathTransformer
from math_tokenizer import MathTokenizer
from expression_evaluator import ExpressionEvaluator
from value_storage import ValueStorage

class MathLLM:
    def __init__(self, config_path: str = "config.yaml", checkpoint_path: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self._setup_device()
        self.tokenizer = MathTokenizer()
        self.model = self._create_model()
        self.evaluator = ExpressionEvaluator()
        self.storage = ValueStorage("inference_values.db")

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _setup_device(self):
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['device_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def _create_model(self):
        model = MathTransformer(
            vocab_size=self.config['model']['vocab_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_attention_heads'],
            max_position_embeddings=self.config['model']['max_position_embeddings'],
            dropout=self.config['model']['dropout']
        )
        return model.to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'tokenizer_variables' in checkpoint:
                self.tokenizer.variables = checkpoint['tokenizer_variables']
            print(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")

    def generate_expression(
        self,
        input_text: str = "",
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        self.model.eval()

        if input_text.strip():
            input_ids = self.tokenizer.encode(input_text, max_length=64)
        else:
            input_ids = torch.tensor([self.tokenizer.special_tokens['<BOS>']], dtype=torch.long)

        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=self.tokenizer.special_tokens['<EOS>']
            )

        generated_text = self.tokenizer.decode(generated[0])
        return generated_text

    def process_expression(self, expression: str) -> Dict[str, Any]:
        expression = expression.strip()

        try:
            success, result, message = self.evaluator.parse_and_evaluate(expression)

            if success and '=' in expression:
                var_name = expression.split('=')[0].strip()
                if isinstance(result, (int, float)):
                    self.storage.set(var_name, result)

                    for var, val in self.evaluator.list_variables().items():
                        if isinstance(val, (int, float)):
                            self.storage.set(var, val)

            return {
                'expression': expression,
                'success': success,
                'result': result,
                'message': message,
                'variables': self.evaluator.list_variables()
            }

        except Exception as e:
            return {
                'expression': expression,
                'success': False,
                'result': None,
                'message': f"Error: {str(e)}",
                'variables': self.evaluator.list_variables()
            }

    def interactive_session(self):
        print("=== Math LLM Interactive Session ===")
        print("Commands:")
        print("  /help     - Show this help")
        print("  /vars     - List all variables")
        print("  /clear    - Clear all variables")
        print("  /storage  - Show storage statistics")
        print("  /generate - Generate a mathematical expression")
        print("  /quit     - Exit the session")
        print("  /exit     - Exit the session")
        print("\nEnter mathematical expressions like: a = 1 + 2")
        print("-" * 50)

        while True:
            try:
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['/quit', '/exit']:
                    print("Goodbye!")
                    break

                elif user_input.lower() == '/help':
                    print("\nAvailable commands:")
                    print("  /help     - Show this help")
                    print("  /vars     - List all variables")
                    print("  /clear    - Clear all variables")
                    print("  /storage  - Show storage statistics")
                    print("  /generate - Generate a mathematical expression")
                    print("  /quit     - Exit the session")

                elif user_input.lower() == '/vars':
                    variables = self.evaluator.list_variables()
                    if variables:
                        print("\nCurrent variables:")
                        for name, value in variables.items():
                            print(f"  {name} = {value}")
                    else:
                        print("No variables defined.")

                elif user_input.lower() == '/clear':
                    self.evaluator.clear_variables()
                    self.storage.clear_all()
                    print("All variables cleared.")

                elif user_input.lower() == '/storage':
                    stats = self.storage.get_statistics()
                    print(f"\nStorage Statistics:")
                    print(f"  Total variables: {stats['total_variables']}")
                    print(f"  Total accesses: {stats['total_accesses']}")
                    if stats['most_accessed']:
                        print(f"  Most accessed: {stats['most_accessed']}")

                elif user_input.lower() == '/generate':
                    context = " ; ".join([f"{k} = {v}" for k, v in self.evaluator.list_variables().items()])
                    generated = self.generate_expression(context)
                    print(f"Generated: {generated}")

                else:
                    result = self.process_expression(user_input)
                    if result['success']:
                        print(f"SUCCESS: {result['message']}")
                    else:
                        print(f"ERROR: {result['message']}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def batch_process(self, expressions: List[str]) -> List[Dict[str, Any]]:
        results = []
        for expr in expressions:
            result = self.process_expression(expr)
            results.append(result)
        return results

def main():
    parser = argparse.ArgumentParser(description="Math LLM - A specialized LLM for mathematical expressions")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--expression", type=str, help="Single expression to evaluate")
    parser.add_argument("--generate", action="store_true", help="Generate a mathematical expression")
    parser.add_argument("--interactive", action="store_true", default=True, help="Start interactive session")

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file {args.config} not found!")
        sys.exit(1)

    llm = MathLLM(args.config, args.checkpoint)

    if args.expression:
        result = llm.process_expression(args.expression)
        if result['success']:
            print(f"SUCCESS: {result['message']}")
        else:
            print(f"ERROR: {result['message']}")

    elif args.generate:
        generated = llm.generate_expression()
        print(f"Generated: {generated}")

    elif args.interactive:
        llm.interactive_session()

if __name__ == "__main__":
    main()