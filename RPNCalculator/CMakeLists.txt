cmake_minimum_required(VERSION 3.15)

project(RPNCalculator)

# Create the executable
add_executable(RPNCalculator
    main.cpp
)

# Link imgui library
target_link_libraries(RPNCalculator PRIVATE imgui)

# Set Windows subsystem
set_target_properties(RPNCalculator PROPERTIES
    WIN32_EXECUTABLE TRUE
)