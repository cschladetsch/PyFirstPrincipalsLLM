#include <iostream>
#include <string>
#include <vector>
#include <sstream>

int main(int argc, char* argv[]) {
    std::cout << "Console Application" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << std::endl;

    std::cout << "Command line arguments:" << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "  argv[" << i << "] = " << argv[i] << std::endl;
    }
    std::cout << std::endl;

    // Interactive console loop
    std::string input;
    std::cout << "Enter commands (type 'exit' or 'quit' to exit):" << std::endl;

    while (true) {
        std::cout << "> ";
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

        // Echo the command
        if (input == "help") {
            std::cout << "Available commands:" << std::endl;
            std::cout << "  help  - Show this help message" << std::endl;
            std::cout << "  echo <text> - Echo back the text" << std::endl;
            std::cout << "  info  - Display system information" << std::endl;
            std::cout << "  exit/quit - Exit the application" << std::endl;
        }
        else if (input.substr(0, 5) == "echo ") {
            std::cout << input.substr(5) << std::endl;
        }
        else if (input == "info") {
            std::cout << "System Information:" << std::endl;
            std::cout << "  Application: ConsoleApp" << std::endl;
            std::cout << "  Build: Release" << std::endl;
            #ifdef _WIN32
            std::cout << "  Platform: Windows" << std::endl;
            #elif __linux__
            std::cout << "  Platform: Linux" << std::endl;
            #elif __APPLE__
            std::cout << "  Platform: macOS" << std::endl;
            #else
            std::cout << "  Platform: Unknown" << std::endl;
            #endif
        }
        else {
            std::cout << "Unknown command: " << input << std::endl;
            std::cout << "Type 'help' for available commands." << std::endl;
        }
    }

    return 0;
}