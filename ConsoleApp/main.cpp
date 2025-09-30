#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include "../external/rang/include/rang.hpp"

// RPN Calculator Stack
std::vector<double> stack;

void printStack() {
    if (stack.empty()) {
        std::cout << rang::fg::yellow << "Stack: (empty)" << rang::fg::reset << std::endl;
        return;
    }
    std::cout << rang::fg::cyan << "Stack: " << rang::fg::reset;
    for (size_t i = 0; i < stack.size(); ++i) {
        std::cout << rang::fg::green << stack[i] << rang::fg::reset;
        if (i < stack.size() - 1) std::cout << " ";
    }
    std::cout << std::endl;
}

void push(double value) {
    stack.push_back(value);
}

double pop() {
    if (stack.empty()) {
        std::cout << rang::fg::red << "Error: Stack underflow" << rang::fg::reset << std::endl;
        return 0.0;
    }
    double value = stack.back();
    stack.pop_back();
    return value;
}

void performOperation(const std::string& op) {
    if (stack.size() < 2) {
        std::cout << rang::fg::red << "Error: Not enough operands on stack" << rang::fg::reset << std::endl;
        return;
    }

    double b = pop();
    double a = pop();

    if (op == "+") {
        push(a + b);
    } else if (op == "-") {
        push(a - b);
    } else if (op == "*") {
        push(a * b);
    } else if (op == "/") {
        if (b == 0.0) {
            std::cout << rang::fg::red << "Error: Division by zero" << rang::fg::reset << std::endl;
            push(a);
            push(b);
        } else {
            push(a / b);
        }
    } else if (op == "^" || op == "pow") {
        push(std::pow(a, b));
    } else {
        std::cout << rang::fg::red << "Error: Unknown operator: " << op << rang::fg::reset << std::endl;
        push(a);
        push(b);
    }
}

void processToken(const std::string& token) {
    // Check if it's a number
    try {
        size_t pos;
        double value = std::stod(token, &pos);
        if (pos == token.length()) {
            push(value);
            return;
        }
    } catch (...) {}

    // Check if it's an operator
    if (token == "+" || token == "-" || token == "*" || token == "/" || token == "^" || token == "pow") {
        performOperation(token);
    } else if (token == "clear" || token == "c") {
        stack.clear();
        std::cout << rang::fg::yellow << "Stack cleared" << rang::fg::reset << std::endl;
    } else if (token == "pop") {
        if (!stack.empty()) {
            std::cout << rang::fg::magenta << "Popped: " << pop() << rang::fg::reset << std::endl;
        } else {
            std::cout << rang::fg::red << "Error: Stack is empty" << rang::fg::reset << std::endl;
        }
    } else if (token == "dup") {
        if (!stack.empty()) {
            push(stack.back());
        } else {
            std::cout << rang::fg::red << "Error: Stack is empty" << rang::fg::reset << std::endl;
        }
    } else if (token == "swap") {
        if (stack.size() >= 2) {
            double a = pop();
            double b = pop();
            push(a);
            push(b);
        } else {
            std::cout << rang::fg::red << "Error: Not enough elements to swap" << rang::fg::reset << std::endl;
        }
    } else if (token == "sqrt") {
        if (!stack.empty()) {
            double a = pop();
            if (a < 0) {
                std::cout << rang::fg::red << "Error: Cannot take square root of negative number" << rang::fg::reset << std::endl;
                push(a);
            } else {
                push(std::sqrt(a));
            }
        } else {
            std::cout << rang::fg::red << "Error: Stack is empty" << rang::fg::reset << std::endl;
        }
    } else {
        std::cout << rang::fg::red << "Error: Unknown token: " << token << rang::fg::reset << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string input;

    while (true) {
        std::cout << rang::fg::blue << "> " << rang::fg::reset;
        if (!std::getline(std::cin, input)) {
            break;
        }

        // Trim whitespace
        input.erase(0, input.find_first_not_of(" \t\n\r"));
        input.erase(input.find_last_not_of(" \t\n\r") + 1);

        if (input.empty()) {
            continue;
        }

        // Check for exit commands
        if (input == "exit" || input == "quit") {
            std::cout << "Goodbye!" << std::endl;
            break;
        }

        // Help command
        if (input == "help") {
            std::cout << "RPN Calculator Commands:" << std::endl;
            std::cout << "  Numbers: Push onto stack" << std::endl;
            std::cout << "  +, -, *, /: Basic operations" << std::endl;
            std::cout << "  ^, pow: Power operation" << std::endl;
            std::cout << "  sqrt: Square root" << std::endl;
            std::cout << "  dup: Duplicate top of stack" << std::endl;
            std::cout << "  swap: Swap top two stack elements" << std::endl;
            std::cout << "  pop: Remove top element" << std::endl;
            std::cout << "  clear, c: Clear stack" << std::endl;
            std::cout << "  stack, s: Show stack" << std::endl;
            std::cout << "  exit, quit: Exit calculator" << std::endl;
            continue;
        }

        // Show stack command
        if (input == "stack" || input == "s") {
            printStack();
            continue;
        }

        // Process input tokens
        std::istringstream iss(input);
        std::string token;
        while (iss >> token) {
            processToken(token);
        }

        // Show result (top of stack)
        if (!stack.empty()) {
            std::cout << rang::fg::yellow << "  = " << rang::style::bold << rang::fg::green << std::fixed << std::setprecision(6) << stack.back() << rang::style::reset << std::endl;
        }
    }

    return 0;
}