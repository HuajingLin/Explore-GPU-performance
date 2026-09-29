cmake_minimum_required(VERSION 3.18)
project(ProgressivePerformanceMatrix LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Usage Instructions
# ---------------------------------------------------------------------------
# If no dedicated GPU is available locally, clone this repository on Colab / Kaggle / AutoDL and run:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75   # T4=75, V100=70, A100=80, 4090=89
#   cmake --build build -j
#
# The generated executables are located in build/, e.g., build/v0_naive
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to T4 (most common on Colab/Kaggle) if not explicitly specified; override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Default to -O3 for Release builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Enable ptxas register usage reporting; uncomment to enable:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

include_directories(${CMAKE_SOURCE_DIR}/src)

# ---------------------------------------------------------------------------
# V0: Naive Baseline
# ---------------------------------------------------------------------------
add_executable(v0_naive src/v0_naive.cu)
set_target_properties(v0_naive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# V1: Shared Memory Tiling
# Parameterized using the TILE_SIZE macro; generates versions for tile sizes
# 8, 16, and 32 simultaneously to facilitate tile size sensitivity sweeps
# without requiring manual code modification and recompilation.
# ---------------------------------------------------------------------------
foreach(TILE 8 16 32)
add_executable(v1_tile${TILE} src/v1_shared_tiling.cu)
target_compile_definitions(v1_tile${TILE} PRIVATE TILE_SIZE=${TILE})
set_target_properties(v1_tile${TILE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()



message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")