cmake_minimum_required(VERSION 3.15)

project(ConsoleApp)

# Create the console executable
add_executable(ConsoleApp
    main.cpp
)

# Add rang include directory
target_include_directories(ConsoleApp PRIVATE
    ${CMAKE_SOURCE_DIR}/external/rang/include
)

# No special Windows subsystem flag for console apps