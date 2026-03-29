cmake_minimum_required(VERSION 3.18)
project(neurobit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Include directories
include_directories(include)

# Core library
add_library(neurobit_core SHARED
    src/neurobit.cpp
)

target_include_directories(neurobit_core PUBLIC include)

# Python module
pybind11_add_module(neurobit
    python/bindings.cpp
)
target_link_libraries(neurobit PRIVATE neurobit_core)

# Option for AVX/SIMD
if(APPLE)
    # On Apple, we check if it is explicitly arm64 or if it's NOT explicitly x86_64
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR NOT CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        if (CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
             message(STATUS "Native compilation for Apple Silicon (ARM64)")
        else()
             message(STATUS "Skipping AVX2 (Default Mac behavior or ARM)")
        endif()
    else()
        target_compile_options(neurobit_core PRIVATE -mavx2)
        message(STATUS "Enabling AVX2 for x86_64 macOS")
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i386")
        target_compile_options(neurobit_core PRIVATE -mavx2)
        message(STATUS "Enabling AVX2 for x86_64 Linux/Windows")
    else()
        message(STATUS "Skipping AVX2 (Non x86 system or ARM: ${CMAKE_SYSTEM_PROCESSOR})")
    endif()
endif()
