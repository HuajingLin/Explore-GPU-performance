# Progressive Performance Matrix on GPU

A progressive CUDA GEMM (General Matrix Multiply) optimization project: starting with a single base compute kernel, five versions are developed—each introducing exactly one specific optimization technique—to demonstrate through quantifiable metrics "why" and "where" the performance improves at each step.   
    
[Project WIKI](https://github.com/HuajingLin/Explore-GPU-performance/wiki/V4-WMMA-GEMM)   
    
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

test:
nvcc -O3 -arch=sm_75 -Xptxas -v register_blocking_gemm.cu

# 3. Run a specific version (N is the matrix size; square matrix where M=N=K=N)
./build/v0_naive 1024       # V0 naive
./build/v1_tile32 1024      # V1 v1_tile8, v1_tile16, v1_tile32 based on tile sizes.
./build/v2_reg4x4 1024      # V2 v2_reg4x4 / v2_reg8x8 based on TM/TN. 
./build/v3_vecdbuf 1024     # V3: vectorized + double buffering
./build/v4_wmma 1024        # V4: WMMA Tensor Core (FP16) + cuBLAS comparison

# 4. Run batch benchmarks (execute from the project root directory)
bash bench/bench.sh       # Results are written to results/results.csv
```

## T4 Measured Data
```diff
Step  Kernel                     % of cuBLAS
 V0   naive                      ~9%
 V1   shared_tiling              8.6–9.8%
 V2   register_blocking          7.5–7.7%
 V3   vectorized_dbuf            ~20%
 V4   wmma_tensorcore            ~10%
 V4.1 wmma_blocktile             67–69%
 V4.2 wmma_blocktile             59%
 V4.3 wmma_blocktile             69%
 V4.4 wmma_blocktile             76–77%

Final result
N	    Configuration	% of cuBLAS        ≥ 70%
512	    V4e-128x64	    77.6%	           (not in the target range)
1024	V4e-w64-sprd	77.2%	           ✓
2048	V4e-w64-sprd	72.0%	           ✓
4096	V4e-w64-sprd	80.3%	           ✓
```

## Project layout
```
gpu_perf/
├── CMakeLists.txt        # Build configuration
├── README.md
├── .gitignore
├── src/                  # Kernel source code for each version + common utilities
│   ├── common.cuh        # Common utility functions
│   ├── v0_naive.cu
│   ├── v1_shared_tiling.cu
│   ├── v2_register_blocking.cu
│   ├── v3_vectorized_dbuf.cu
│   └── v4_wmma_tensorcore.cu
├── bench/
│   └── bench.sh          # Batch compilation, scale sweeping, and CSV output
├── results/              # Raw benchmark data (CSV) and summary plots (ignored by git)
└── docs/                 # Nsight Compute screenshots and analysis notes
```