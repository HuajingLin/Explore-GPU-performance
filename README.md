# Progressive Performance Matrix on GPU

A progressive CUDA GEMM (General Matrix Multiply) optimization project: starting with a single base compute kernel, five versions are developed—each introducing exactly one specific optimization technique—to demonstrate through quantifiable metrics "why" and "where" the performance improves at each step.   
    
[project WIKI](https://github.com/HuajingLin/gpu_perf/wiki/Progressive-Performance-Matrix-on-GPU)   
    
## Environment

- Language: CUDA C++ 17; Build System: CMake (>=3.18, requiring CUDA language support)
- Runtime Environment: Local machine lacks a discrete GPU (Intel Iris Xe); CUDA components run on cloud GPUs
- Google Colab (Free T4, architecture `sm_75`)
- Profiling Tools: Nsight Compute (`ncu`), `nvcc --ptxas-options=-v` (to inspect register usage)
- Baseline Comparison: cuBLAS (`cublasSgemm`, integrated at the V4 stage)

## Build and Run

```bash
# 1. Upload the repository to Colab (via git clone)
git clone <your-repo-url> gpu_perf
cd gpu_perf

# 2. Build with CMake (use sm_75 for T4; 80 for A100, 70 for V100, 89 for 4090)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Run a specific version (N is the matrix size; square matrix where M=N=K=N)
./build/v0_naive 1024
./build/v1_tile32 1024      # V1 v1_tile8, v1_tile16, v1_tile32 based on tile sizes 8, 16, and 32 ( ..)
./build/v2_reg4x4 1024      # V2 v2_reg4x4 / v2_reg8x8 based on TM/TN. 

# 4. Run batch benchmarks (execute from the project root directory)
cd ..
bash bench/bench.sh       # Results are written to results/results.csv
```

## T4 Measured Data
version,tile_size,N,   avg_ms, gflops
V0,     NA,      1024, 6.114,  351.23
V1,     32,      1024, 2.690,  798.25
best speedup: 2.27x

## Project layout

```
gpu_perf/
├── CMakeLists.txt        # Build configuration
├── .gitignore
├── src/                  # Kernel source code for each version + common utilities
│   ├── common.cuh        # Common utility functions
│   ├── v0_naive.cu
│   ├── v1_shared_tiling.cu
    └── v2_register_blocking.cu
├── bench/
│   └── bench.sh          # Batch compilation, scale sweeping, and CSV output
├── results/               # Raw benchmark data (CSV) and summary plots (ignored by git; generated locally/in the cloud)
└── docs/                 # Nsight Compute screenshots and analysis notes
```