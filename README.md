# PyLLM Project

A transformer-based language model that **learns mathematics from scratch** through self-training, plus C++ RPN calculator applications.

## What Makes This Special

This project features a **self-learning mathematical AI** that discovers arithmetic patterns through training rather than using hardcoded rules. The transformer model learns to:
- Understand and evaluate mathematical expressions
- Solve equations with variables
- Remember assigned values across sessions
- Perform arithmetic operations it was never explicitly programmed to do

It learns math the same way large language models learn language - through pattern recognition in training data.

## Python Components

### Core Math LLM
- **Training Pipeline** (`train.py`): Full training loop with mixed precision, gradient clipping, TensorBoard logging, and checkpoint management
- **Inference Engine** (`math_llm.py`): Interactive session support with variable persistence and expression evaluation
- **Transformer Model** (`math_transformer.py`): Custom transformer architecture optimized for mathematical expressions
- **Tokenizer** (`math_tokenizer.py`): Specialized tokenizer for math syntax including operators, functions, and variables
- **Expression Evaluator** (`expression_evaluator.py`): Parse and evaluate mathematical expressions with variable support
- **Data Generator** (`data_generator.py`): Synthetic training data generation for mathematical expressions
- **Value Storage** (`value_storage.py`): SQLite-based persistent storage for variables and statistics

### Getting Started with Python

1. **Install Dependencies**:
```bash
pip install torch pyyaml tqdm tensorboard numpy
```

2. **Configure Training** (edit `config.yaml`):
```yaml
model:
  hidden_size: 512
  num_layers: 6
  num_attention_heads: 8
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
```

3. **Train the Model**:
```bash
python train.py
```

4. **Run Interactive Session**:
```bash
python math_llm.py --interactive
```

5. **Example Usage**:
```python
>>> a = 5
SUCCESS: a = 5.0
>>> b = a * 2
SUCCESS: b = 10.0
>>> c = sqrt(a^2 + b^2)
SUCCESS: c = 11.180339887498949
>>> /vars
Current variables:
  a = 5.0
  b = 10.0
  c = 11.180339887498949
```

### Python Features

- **Self-Learning**: Transformer model learns mathematical patterns from generated data
- **Interactive REPL**: Command-line interface with persistent variable storage
- **Expression Evaluation**: Supports arithmetic, trigonometric, and algebraic operations
- **Batch Processing**: Process multiple expressions programmatically
- **Model Checkpointing**: Save and resume training with full state persistence
- **TensorBoard Integration**: Monitor training metrics in real-time
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision

## C++ Components

### RPN Calculators
- **RPN Calculator (GUI)**: Windows application with ImGui interface for Reverse Polish Notation calculations
- **RPN Calculator (Console)**: Colorful command-line RPN calculator with rang.hpp
- **CMake Build System**: Modern CMake configuration with organized build structure
- **External Libraries**: ImGui (with DirectX11 backend) and rang for colors

## Project Structure

```
PyLLM/
├── Python Components
│   ├── train.py                    # Training pipeline
│   ├── math_llm.py                 # Inference and interactive session
│   ├── math_transformer.py         # Transformer model architecture
│   ├── math_tokenizer.py           # Mathematical expression tokenizer
│   ├── expression_evaluator.py     # Expression parsing and evaluation
│   ├── data_generator.py           # Training data generation
│   ├── value_storage.py            # Variable persistence (SQLite)
│   ├── example_usage.py            # Usage examples
│   ├── config.yaml                 # Training configuration
│   └── requirements.txt            # Python dependencies
│
├── C++ Components
│   ├── Bin/                        # All executables output here
│   ├── RPNCalculator/              # Windows GUI RPN calculator
│   │   ├── CMakeLists.txt
│   │   └── main.cpp
│   ├── ConsoleApp/                 # Console RPN calculator
│   │   ├── CMakeLists.txt
│   │   └── main.cpp
│   └── external/                   # External dependencies
│       ├── imgui/                  # ImGui submodule
│       └── rang/                   # rang.hpp for colors
│
└── CMakeLists.txt                  # Root CMake configuration
```

## Building C++ Components

### Prerequisites

- CMake 3.15 or higher
- Visual Studio 2022 (or compatible C++ compiler)
- Git (for submodules)

### Build Instructions

1. Clone the repository:
```bash
git clone https://github.com/cschladetsch/PyFirstPrincipalsLLM.git
cd PyLLM
```

2. Initialize submodules:
```bash
git submodule update --init --recursive
```

3. Build the project:
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

All executables will be in the `Bin/` directory.

## C++ Applications

### ConsoleApp - RPN Calculator (Console)

A colorful console-based RPN calculator with intuitive color-coded output.

**Run:**
```bash
./Bin/ConsoleApp.exe
```

**Features:**
- **Colorful Interface**: Uses rang.hpp for color-coded output
  - Blue prompt
  - Green numbers
  - Red errors
  - Yellow info messages
- **RPN Operations**: Standard reverse polish notation
- **Basic operators**: `+`, `-`, `*`, `/`, `^` (power)
- **Stack operations**: `dup`, `swap`, `pop`, `clear`
- **Math functions**: `sqrt`
- **Commands**: `help`, `stack`/`s`, `exit`/`quit`

**Example Usage:**
```
> 3 4 +
  = 7.000000
> 5 2 ^
  = 25.000000
> 16 sqrt
  = 4.000000
> stack
Stack: 7 25 4
```

### RPNCalculator - RPN Calculator (GUI)

A Windows GUI application with ImGui interface for RPN calculations.

**Run:**
```bash
./Bin/RPNCalculator.exe
```

**Features:**
- Visual stack display
- Number pad (0-9, decimal point)
- Basic operators (+, -, *, /)
- Interactive buttons
- Text input field
- Clear and Pop buttons
- DirectX11 rendering

## RPN (Reverse Polish Notation)

RPN is a mathematical notation where operators follow their operands. For example:
- Infix: `3 + 4` → RPN: `3 4 +`
- Infix: `(5 + 3) * 2` → RPN: `5 3 + 2 *`

### How It Works

1. Enter numbers to push them onto the stack
2. Enter operators to perform operations on the top stack elements
3. The result stays on the stack for further operations

## Technologies Used

- **C++17**: Modern C++ with STL
- **CMake**: Cross-platform build system
- **ImGui**: Immediate mode GUI library for the GUI calculator
- **DirectX11**: Graphics API for Windows rendering
- **rang.hpp**: Header-only library for terminal colors

## License

MIT License