// common.cuh
// Common utility functions
/* Includes: CUDA error checking, random matrix initialization, 
CPU reference GEMM, correctness verification, 
and GFLOPS calculation.*/

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------- CUDA Error-checking macros ----------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// -------Random matrix initialization (uniform distribution over [-1, 1])------
inline void init_matrix(float* mat, int rows, int cols, unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }
}

// ---------- CPU reference implementation C = A(MxK) * B(KxN) ----------
// For correctness verification only; a size of ≤ 512 is recommended, 
// otherwise it will be very slow.
inline void cpu_gemm_ref(const float* A, const float* B, float* C, int M,
                          int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---------- Correctness check (allclose, tolerance 1e-3)----------
inline bool verify_result(const float* gpu_result, const float* cpu_ref,
                           int size, float atol = 1e-3f, float rtol = 1e-3f) {
    double max_abs_err = 0.0;
    int first_bad_idx = -1;
    for (int i = 0; i < size; ++i) {
        double diff = std::fabs(gpu_result[i] - cpu_ref[i]);
        double tol = atol + rtol * std::fabs(cpu_ref[i]);
        if (diff > tol) {
            if (first_bad_idx < 0) first_bad_idx = i;
        }
        if (diff > max_abs_err) max_abs_err = diff;
    }
    printf("  [verify] max_abs_err=%.6e%s\n", max_abs_err,
           first_bad_idx < 0 ? " -> PASS" : " -> FAIL");
    if (first_bad_idx >= 0) {
        printf("  [verify] first mismatch at idx=%d gpu=%.6f cpu=%.6f\n",
               first_bad_idx, gpu_result[first_bad_idx], cpu_ref[first_bad_idx]);
    }
    return first_bad_idx < 0;
}

// ---------- GFLOPS Calculation ----------
inline double compute_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * M * N * K;
    double seconds = ms / 1000.0;
    return (flops / seconds) / 1e9;
}

// ---------- GPU timing: Run a kernel multiple times and calculate the average. ----------
template <typename KernelLauncher>
float benchmark_kernel(KernelLauncher launch_fn, int warmup_iters = 3,
                        int bench_iters = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < warmup_iters; ++i) launch_fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; ++i) launch_fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));//Wait for the 'stop' event to complete on the GPU

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / bench_iters;  // Average time per instance (ms)
}
