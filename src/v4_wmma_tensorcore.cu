// v4_wmma_tensorcore.cu

#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, \
            __LINE__, (int)status);                                         \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------- Convert FP32 matrix to FP16 (Host-side) ----------
void convert_to_half(const float* src, half* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

// ---------- V4 Basic WMMA Tensor Core Kernel ----------
// Each warp is responsible for a WMMA_M x WMMA_N output tile in C,
// accumulating WMMA_K elements at a time along the K dimension.
// Reads A and B directly from global memory (without shared memory reuse);
// this is the simplest version that successfully executes the Tensor Core path.
__global__ void wmma_naive_gemm_kernel(const half* __restrict__ A,
                                       const half* __restrict__ B,
                                            float* __restrict__ C, 
                                            int M, int N, int K) {
    // Global warp coordinates: each warp computes a 16x16 output tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
    wmma::row_major>
    a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
    wmma::row_major>
    b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;

    if (aRow < M && bCol < N) {
        for (int k = 0; k < K; k += WMMA_K) {
            // load_matrix_sync reads a tile of WMMA_M(or K) x WMMA_K(or N) data;
            // M, N, and K must be multiples of 16, otherwise boundary writes occur
            // (checks are performed on the host side)
            wmma::load_matrix_sync(a_frag, A + (size_t)aRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + (size_t)k * N + bCol, N); 
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(C + (size_t)aRow * N + bCol, acc_frag, N, wmma::mem_row_major);
    }
}

void run_wmma_gemm(const half* d_A, const half* d_B, float* d_C, int M, int N, int K) {
    // blockDim.x=128 => 128/32=4 warps along the M-dimension; blockDim.y=4 => 4 warps along the N-dimension
    // Each block has a total of 16 warps, responsible for a (4*WMMA_M) x (4*WMMA_N) = 64x64 output tile
    dim3 blockDim(128, 4);
    dim3 gridDim((M + (WMMA_M * (blockDim.x / 32)) - 1) /
    (WMMA_M * (blockDim.x / 32)),
    (N + (WMMA_N * blockDim.y) - 1) / (WMMA_N * blockDim.y));
    wmma_naive_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ---------- cuBLAS Comparison Benchmark: FP16 Input + FP32 Computation + Tensor Core Path ----------
// Note: cuBLAS uses column-major storage, whereas our matrices are row-major. Leveraging a classic technique:
// Convert the row-major operation C(MxN) = A(MxK) * B(KxN) into an equivalent
// column-major operation: C^T(NxM) = B^T(NxK) * A^T(KxM). We treat our row-major
// matrices as transposed column-major matrices (without actually performing
// a transposition); we simply swap the M and N parameters and the leading
// dimensions—this is the standard approach in the CUDA ecosystem for
// processing row-major data using cuBLAS. 
void run_cublas_gemm(cublasHandle_t handle, 
                    const half* d_A, const half* d_B, float* d_C, 
                    int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N,
    d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = N, K = N;

    printf("=== V4 WMMA Tensor Core GEMM (FP16 input, FP32 accumulation) ===\n");
    printf("M=%d N=%d K=%d\n", M, N, K);

    // ---- Pre-check: WMMA requires M, N, and K to be multiples of 16 ----
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        fprintf(stderr, "ERROR: M, N, and K must be divisible by 16 (WMMA tile requirement)\n");
        return EXIT_FAILURE;
    }

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    // ---- Host side: generate FP32 data and convert to FP16 ----
    float *h_A_f32, *h_B_f32, *h_C_wmma, *h_C_cublas;
    h_A_f32 = (float*)malloc(size_A * sizeof(float));
    h_B_f32 = (float*)malloc(size_B * sizeof(float));
    h_C_wmma = (float*)malloc(size_C * sizeof(float));
    h_C_cublas = (float*)malloc(size_C * sizeof(float));

    init_matrix(h_A_f32, M, K, 42);
    init_matrix(h_B_f32, K, N, 24);

    half *h_A_f16, *h_B_f16;
    h_A_f16 = (half*)malloc(size_A * sizeof(half));
    h_B_f16 = (half*)malloc(size_B * sizeof(half));
    convert_to_half(h_A_f32, h_A_f16, size_A);
    convert_to_half(h_B_f32, h_B_f16, size_B);

    // ---- device side// Allocate device memory ----
    half *d_A, *d_B;
    float *d_C_wmma, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_f16, size_A * sizeof(half),
    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_f16, size_B * sizeof(half),
    cudaMemcpyHostToDevice));

    // ---------- Correctness verification: Note that tolerance must be relaxed for FP16 precision;
    // the 1e-3 tolerance used in previous versions is not applicable here ----------
    if (N <= 512) {
        run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_wmma, d_C_wmma, size_C * sizeof(float),
        cudaMemcpyDeviceToHost));

        float* h_C_ref = (float*)malloc(size_C * sizeof(float));
        cpu_gemm_ref(h_A_f32, h_B_f32, h_C_ref, M, N, K);
        // FP16 precision is limited, so tolerance is relaxed to atol=0.5, rtol=5%
        // (significantly looser than the 1e-3 used previously;
        // this is due to the inherent precision of FP16 inputs, not a kernel implementation error)
        printf("  [verify wmma kernel] (Note: FP16 precision; tolerances relaxed to atol=0.5/rtol=5%)\n");
        verify_result(h_C_wmma, h_C_ref, (int)size_C, 0.5f, 0.05f);
        free(h_C_ref);
    } else {
        printf("  [verify] N > 512; skipping full CPU verification (too slow). Suggest verifying correctness separately using a smaller scale.\n");
    }

    // ---------- Benchmark: Our WMMA kernel ----------
    float avg_ms_wmma = benchmark_kernel([&]() { run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K); });
    double gflops_wmma = compute_gflops(M, N, K, avg_ms_wmma);
    printf("  [bench] Custom WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms_wmma, gflops_wmma);

    // ---------- Benchmark: cuBLAS baseline (also using Tensor Core path) ----------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float avg_ms_cublas = benchmark_kernel([&]() {
        run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    });

    double gflops_cublas = compute_gflops(M, N, K, avg_ms_cublas);
    printf("  [bench] cuBLAS (TensorCore): avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms_cublas, gflops_cublas);

    double pct_of_cublas = 100.0 * gflops_wmma / gflops_cublas;
    printf(" [compare] Our kernel achieved %.1f%% of cuBLAS performance\n", pct_of_cublas);
    printf("  (Acceptance criteria: >= 70%% for N>=1024 is considered passing)\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaFree(d_C_cublas));
    free(h_A_f32);
    free(h_B_f32);
    free(h_A_f16);
    free(h_B_f16);
    free(h_C_wmma);
    free(h_C_cublas);

    return 0;
}