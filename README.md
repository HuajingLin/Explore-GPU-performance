# Progressive Performance Matrix on GPU

A progressive CUDA GEMM (General Matrix Multiply) optimization project: starting with a single base compute kernel, five versions are developed—each introducing exactly one specific optimization technique—to demonstrate through quantifiable metrics "why" and "where" the performance improves at each step.

## Environment

- Language: CUDA C++ 17; Build System: CMake (>=3.18, requiring CUDA language support)
- Runtime Environment: Local machine lacks a discrete GPU (Intel Iris Xe); CUDA components run on cloud GPUs
- Google Colab (Free T4, architecture `sm_75`)
- Profiling Tools: Nsight Compute (`ncu`), `nvcc --ptxas-options=-v` (to inspect register usage)
- Baseline Comparison: cuBLAS (`cublasSgemm`, integrated at the V4 stage)

## Build and Run

```bash
# 1. Upload the repository to Colab (via git clone or by uploading a zip file)
git clone <your-repo-url> gpu_perf
cd gpu_perf

# 2. Build with CMake (use sm_75 for T4; 80 for A100, 70 for V100, 89 for 4090)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Run a specific version (N is the matrix size; square matrix where M=N=K=N)
./v0_naive 1024
./v1_tile32 1024      # V1 generates separate executables for tile sizes 8, 16, and 32 (e.g., v1_tile8, v1_tile16, v1_tile32)

# 4. Run batch benchmarks (execute from the project root directory)
cd ..
bash bench/bench.sh       # Results are written to results/results.csv
```



## Project layout

```
gpu_perf/
├── CMakeLists.txt        # Build configuration
├── .gitignore
├── src/                  # Kernel source code for each version + common utilities
│   ├── common.cuh        # Common utility functions
│   ├── v0_naive.cu
│   └── v1_shared_tiling.cu
├── bench/
│   └── bench.sh          # Batch compilation, scale sweeping, and CSV output
├── results/               # Raw benchmark data (CSV) and summary plots (ignored by git; generated locally/in the cloud)
└── docs/                 # Nsight Compute screenshots and analysis notes
```