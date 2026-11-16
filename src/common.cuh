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
#include <algorithm>
#include <vector>

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
// For correctness verification only; a size of <= 512 is recommended, 
// otherwise it will be very slow.
inline void cpu_gemm_ref(const float* A, const float* B, float* C, int M,
                          int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Accumulate in double so the reference itself carries no
            // visible rounding error; the measured error is then the GPU's.
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

// ---------- Correctness check (allclose, tolerance 1e-3)----------
inline bool verify_result(const float* gpu_result, const float* cpu_ref,
                           int size, float atol = 1e-3f, float rtol = 1e-3f) {
    double max_abs_err = 0.0;
    int first_bad_idx = -1;
    int nan_count = 0;
    for (int i = 0; i < size; ++i) {
        double diff = std::fabs(static_cast<double>(gpu_result[i]) -
                                static_cast<double>(cpu_ref[i]));
        double tol = atol + rtol * std::fabs(static_cast<double>(cpu_ref[i]));
        // Written as !(diff <= tol) on purpose: a NaN diff (NaN in either
        // input, or inf - inf) compares false with everything, so
        // "diff > tol" would let it pass.
        if (!(diff <= tol)) {
            if (first_bad_idx < 0) first_bad_idx = i;
        }
        if (std::isnan(diff)) {
            ++nan_count;  // kept out of the maximum, reported separately
        } else if (diff > max_abs_err) {
            max_abs_err = diff;
        }
    }
    printf("  [verify] max_abs_err=%.6e%s\n", max_abs_err,
           first_bad_idx < 0 ? " -> PASS" : " -> FAIL");
    if (nan_count > 0) {
        printf("  [verify] %d element(s) compare as NaN (NaN in a result, or "
               "inf - inf)\n",
               nan_count);
    }
    if (first_bad_idx >= 0) {
        printf("  [verify] first mismatch at idx=%d gpu=%.6f cpu=%.6f\n",
               first_bad_idx, gpu_result[first_bad_idx], cpu_ref[first_bad_idx]);
    }
    return first_bad_idx < 0;
}

// ---------- GFLOPS Calculation: GEMM  2*M*N*K ----------
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
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / bench_iters;  // Average time per instance (ms)
}

// ---------- Repeated multiple times within the same measurement window ----------
// Repeats the SAME short window `repeats` times and reports median / min /
// max of the per-launch time. The window length is unchanged on purpose:
// on a power-capped GPU a longer window lets the clock adapt to the kernel
// and measures something else (see the [sustained] section of the benchmark).
struct BenchStats {
    float median_ms;
    float min_ms;
    float max_ms;
};

template <typename KernelLauncher>
BenchStats benchmark_kernel_stats(KernelLauncher launch_fn, int repeats = 5,
                                   int warmup_iters = 3, int bench_iters = 10) {
    if (repeats < 1) repeats = 1;
    std::vector<float> t;
    for (int r = 0; r < repeats; ++r) {
        // warm up only before the first repeat
        t.push_back(benchmark_kernel(launch_fn, r == 0 ? warmup_iters : 0,
                                     bench_iters));
    }
    std::sort(t.begin(), t.end());
    const size_t n = t.size();
    BenchStats st;
    st.median_ms = (n % 2) ? t[n / 2] : 0.5f * (t[n / 2 - 1] + t[n / 2]);
    st.min_ms = t.front();
    st.max_ms = t.back();
    return st;
}
