# PyLLM Project

A C++ project featuring multiple applications built with CMake, including a Windows GUI RPN calculator and a colorful console RPN calculator.

## Features

- **RPN Calculator (GUI)**: Windows application with ImGui interface for Reverse Polish Notation calculations
- **RPN Calculator (Console)**: Colorful command-line RPN calculator with rang.hpp
- **CMake Build System**: Modern CMake configuration with organized build structure
- **External Libraries**: ImGui (with DirectX11 backend) and rang for colors

## Project Structure

```
PyLLM/
├── Bin/                      # All executables output here
├── RPNCalculator/            # Windows GUI RPN calculator
│   ├── CMakeLists.txt
│   └── main.cpp
├── ConsoleApp/               # Console RPN calculator
│   ├── CMakeLists.txt
│   └── main.cpp
├── external/                 # External dependencies
│   ├── imgui/               # ImGui submodule
│   └── rang/                # rang.hpp for colors
└── CMakeLists.txt           # Root CMake configuration
```

## Building

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

## Applications

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