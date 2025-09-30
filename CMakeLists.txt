cmake_minimum_required(VERSION 3.15)
project(PyLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/Bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/Bin)

# Add imgui submodule (exclude tests)
set(IMGUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/imgui EXCLUDE_FROM_ALL)

# Add subdirectories for executables
add_subdirectory(RPNCalculator)