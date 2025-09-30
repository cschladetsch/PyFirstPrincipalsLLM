cmake_minimum_required(VERSION 3.15)

project(ConsoleApp)

# Create the console executable
add_executable(ConsoleApp
    main.cpp
)

# No special Windows subsystem flag for console apps