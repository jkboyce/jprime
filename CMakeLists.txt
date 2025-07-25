#[[
# CMakeLists.txt
#
# File to configure and build `jprime`, for use with the CMake build system.
#
# Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
#
# This file is distributed under the MIT License.
#

Explanation and how to use
--------------------------
You must have CMake installed on your system, version 3.21 or later. From the
root of the jprime repository you can build by running the following commands:

1. List the available presets for your platform (defined in CMakePresets.json):
   cmake --list-presets

2. Configure one of the presets (in this example the Windows CPU version):
   cmake --preset win-cpu-release

3. Build an executable for the configured preset (choose one):
   cmake --build --preset win-cpu-release
   cmake --build --preset win-cpu-release --clean-first

4. Execute a suite of tests (cd into the correct build folder first):
   ctest

The executable is built at e.g. \build\win-cpu-release\Release\jprime.exe
On Unix-based systems (macOS, Linux), after step 2 you can cd to the relevant
config folder within /build and run `make` there directly, in place of step 3.

]]

cmake_minimum_required(VERSION 3.21)
project(jprime LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directory for the executable to be in the build root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Option to enable CUDA support
option(JPRIME_ENABLE_CUDA "Enable CUDA support for jprime" OFF)

# Define C++ source files
set(JPRIME_CXX_SOURCES
    src/jprime.cc
    src/jprime_tests.cc
    src/common/Coordinator.cc
    src/common/Graph.cc
    src/common/Pattern.cc
    src/common/SearchConfig.cc
    src/common/SearchContext.cc
    src/common/State.cc
    src/common/WorkAssignment.cc
    src/cpu/CoordinatorCPU.cc
    src/cpu/Worker.cc
    src/cpu/GenLoopsRecursive.cc
    src/cpu/GenLoopsIterative.cc
)

# Base include directories for C++
set(JPRIME_BASE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
)

# Executable target definition
add_executable(jprime)

# Add C++ sources and base include directories
target_sources(jprime PRIVATE ${JPRIME_CXX_SOURCES})
target_include_directories(jprime PRIVATE ${JPRIME_BASE_INCLUDE_DIRS})

# C++ compile flags (non-default)
if(MSVC)
    # MSVC specific C++ compiler flags
    # /W4 is similar to -Wall, /EHsc is for exception handling
    target_compile_options(jprime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /EHsc>)
    # disable MSVC warnings:
    #   C4127 -- conditional expression is constant
    #   C4146 -- unary minus applied to unsigned type
    #   C4996 -- 'ctime' may be unsafe
    target_compile_options(jprime PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4127 /wd4146 /wd4996>)
else()
    # GCC/Clang specific C++ compiler flags
    target_compile_options(jprime PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

if(JPRIME_ENABLE_CUDA)
    message(STATUS "JPRIME_ENABLE_CUDA is ON. Configuring CUDA support.")

    # Set target CUDA architectures:
    # PTX + SASS for 8.9 (RTX 4000 series)
    # PTX + SASS for 12.0 (RTX 5000 series)
    # PTX for 6.0 (everything else)
    set_property(TARGET jprime PROPERTY CUDA_ARCHITECTURES 89 120 60-virtual)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDA Toolkit version ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit bin: ${CUDAToolkit_BIN_DIR}")

    # Add CUDA-specific C++ source files (compiled as CXX with CUDA context)
    target_sources(jprime PRIVATE src/cuda/CoordinatorCUDA.cc)

    # Add CUDA kernel files (.cu)
    target_sources(jprime PRIVATE src/cuda/CudaKernels.cu)

    # Define CUDA_ENABLED for all C++ compilations when CUDA is active
    target_compile_definitions(jprime PRIVATE CUDA_ENABLED)

    # Add CUDA-specific include directories (for both CXX and CU files)
    target_include_directories(jprime PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    )

    # Set CUDA standard for .cu files
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # CUDA-specific compiler flags for .cu files
    set(JPRIME_CUDA_CU_FLAGS "-lineinfo" "-Wno-deprecated-gpu-targets")
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND JPRIME_CUDA_CU_FLAGS "-O3")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND JPRIME_CUDA_CU_FLAGS "-g")
    endif()

    # Host-compiler specific flags for nvcc
    if(MSVC)
        list(APPEND JPRIME_CUDA_CU_FLAGS -Xcompiler "/wd4146")
    endif()

    target_compile_options(jprime PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${JPRIME_CUDA_CU_FLAGS}>
    )

    # Link with CUDA runtime library
    target_link_libraries(jprime PRIVATE CUDA::cudart)

    message(STATUS "CUDA support configured for jprime.")
else()
    message(STATUS
        "JPRIME_ENABLE_CUDA is OFF. Building CPU-only version of jprime.")
endif()

# Testing
enable_testing()
foreach(test_num RANGE 1 17)
    add_test(
        NAME JPrimeTest${test_num}
        COMMAND $<TARGET_FILE:jprime> -test ${test_num}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()

message(STATUS
    "jprime executable will be built in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
