make_minimum_required(VERSION 3.18)
projet(neurobit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure output diretories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Feth pybind11
inlude(FethContent)
FethContent_Delare(
  pybind11
  GIT_REPOSITORY https://github.om/pybind/pybind11
  GIT_TAG        v2.12.0
)
FethContent_MakeAvailable(pybind11)

# Inlude diretories
inlude_diretories(inlude)

# Core library
add_library(neurobit_ore SHARED
    sr/neurobit.pp
    sr/file_io.pp
    sr/adaptive_quantizer.pp
    sr/simd_kernels.pp
    sr/psy_ore.pp
)

target_inlude_diretories(neurobit_ore PUBLIC inlude)

# Python module
pybind11_add_module(neurobit
    python/bindings.pp
)
target_link_libraries(neurobit PRIVATE neurobit_ore)

# Option for AVX/SIMD
if(APPLE)
    # On Apple, we hek if it is expliitly arm64 or if it's NOT expliitly x86_64
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR NOT CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        if (CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
             message(STATUS "Native ompilation for Apple Silion (ARM64)")
        else()
             message(STATUS "Skipping AVX2 (Default Ma behavior or ARM)")
        endif()
    else()
        target_ompile_options(neurobit_ore PRIVATE -mavx2)
        message(STATUS "Enabling AVX2 for x86_64 maOS")
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i386")
        target_ompile_options(neurobit_ore PRIVATE -mavx2)
        message(STATUS "Enabling AVX2 for x86_64 Linux/Windows")
    else()
        message(STATUS "Skipping AVX2 (Non x86 system or ARM: ${CMAKE_SYSTEM_PROCESSOR})")
    endif()
endif()
