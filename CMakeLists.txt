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
You must have CMake installed on your system, version 3.18 or later.

Configure and build (CPU version):
   cd build
   cmake ..
   cmake --build .
   --> This will create the `jprime` executable in the `build` directory.

Configure and build (CUDA version):
   cd build
   cmake .. -DJPRIME_ENABLE_CUDA=ON
   cmake --build .
   --> This will enable CUDA support. CMake will find your CUDA Toolkit. If it's
       not found, CMake will error out.

Clean the build directory:
   cd build
   cmake --build . --target clean

Build types:
   * release build (optimized): `cmake .. -DCMAKE_BUILD_TYPE=Release`
   * debug build (with debug symbols): `cmake .. -DCMAKE_BUILD_TYPE=Debug`

Run tests:
   cd build
   ctest
   --> After building, this runs all the test cases defined below.
]]

cmake_minimum_required(VERSION 3.18)
# Version 3.18 for CMAKE_CUDA_ARCHITECTURES and modern CUDA/C++20 support
project(jprime LANGUAGES CXX)
# CUDA language enabled conditionally below

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directory for the executable to be in the build root
# (e.g., build/jprime)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Option to enable CUDA support
option(JPRIME_ENABLE_CUDA "Enable CUDA support for jprime" OFF)

# Define C++ source files
set(JPRIME_CXX_SOURCES
    src/jprime.cc
    src/jprime_tests.cc
    src/Coordinator.cc
    src/Graph.cc
    src/Pattern.cc
    src/SearchConfig.cc
    src/SearchContext.cc
    src/State.cc
    src/WorkAssignment.cc
    src/cpu/CoordinatorCPU.cc
    src/cpu/Worker.cc
    src/cpu/GenLoopsRecursive.cc
    src/cpu/GenLoopsIterative.cc
)

# Base include directories for C++
set(JPRIME_BASE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu
)

# Executable target definition
add_executable(jprime)

# Add C++ sources and base include directories
target_sources(jprime PRIVATE ${JPRIME_CXX_SOURCES})
target_include_directories(jprime PRIVATE ${JPRIME_BASE_INCLUDE_DIRS})

# Common C++ compile flags (mirroring Makefile's CFLAGS)
set(JPRIME_CXX_FLAGS "-Wall;-Wextra;-Werror;-O3")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND JPRIME_CXX_FLAGS "-g")
endif()
target_compile_options(jprime PRIVATE ${JPRIME_CXX_FLAGS})

# CUDA configuration (if enabled)
if(JPRIME_ENABLE_CUDA)
    message(STATUS "JPRIME_ENABLE_CUDA is ON. Configuring CUDA support.")
    enable_language(CUDA) # Enable CUDA language support for the project

    find_package(CUDAToolkit REQUIRED)
    message(STATUS "Found CUDAToolkit version ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDAToolkit root:")
    message(STATUS "    ${CUDAToolkit_TOOLKIT_ROOT}")

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

    # CUDA-specific compiler flags for .cu files (mirroring makefile's
    # NVCCFLAGS)
    set(JPRIME_CUDA_CU_FLAGS
        "-O3"
        "-lineinfo"
        "-Xcudafe --diag_suppress=68"
        "-Wno-deprecated-gpu-targets"
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND JPRIME_CUDA_CU_FLAGS "-g")
    endif()
    target_compile_options(jprime PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${JPRIME_CUDA_CU_FLAGS}>
    )

    # Set CUDA architectures (mirroring Makefile's -gencode flags)
    # -gencode arch=compute_60,code=compute_60  => PTX for 6.0 + SASS for 6.0
    # -gencode arch=compute_89,code=sm_89      => PTX for 8.9 + SASS for 8.9
    # CMake equivalent: "60+PTX;89+PTX"
    set(CMAKE_CUDA_ARCHITECTURES "60+PTX;89+PTX")

    # Link with CUDA runtime library
    target_link_libraries(jprime PRIVATE CUDA::cudart)

    message(STATUS "CUDA support configured for jprime.")
else()
    message(STATUS
        "JPRIME_ENABLE_CUDA is OFF. Building CPU-only version of jprime.")
endif()


# Testing
enable_testing()
foreach(test_num RANGE 1 16)
    add_test(
        NAME JPrimeTest${test_num}
        COMMAND $<TARGET_FILE:jprime> -test ${test_num}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()

message(STATUS
    "jprime executable will be built in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
