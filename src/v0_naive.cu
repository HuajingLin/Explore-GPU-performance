// v0_naive.cu
// V0: Naive Global Memory Baseline
//
// Each thread computes a single element of the output matrix C by directly reading
// a row of A and a column of B from global memory to calculate their dot product.
// This implementation includes no optimizations and serves as a performance baseline. 
//
// Compile: nvcc -O3 -arch=sm_75 v0_naive.cu -o v0_naive
// (sm_75 corresponds to T4; for Colab/Kaggle, adjust arch based on the actual GPU, e.g., sm_70/sm_80/sm_86)
//
// Run: ./build/v0_naive [N]   (Default N=1024, square matrix M=N=K=N)

#include "common.cuh"

// ---------- Naive GEMM Kernel ----------
// C[M,N] = A[M,K] * B[K,N]
// Each thread is responsible for computing one element of C: C[row][col]
__global__ void naive_gemm_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N,
                                int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Direct read from global memory; no data reuse
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_naive_gemm(const float* d_A, const float* d_B, float* d_C, int M,
                    int N, int K) {
    dim3 block(16, 16); //16 × 16 = 256 threads
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = N, K = N;  // Square matrix test

    printf("=== V0 Naive GEMM ===\n");
    printf("M=%d N=%d K=%d\n", M, N, K);

    size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    init_matrix(h_A, M, K, 42);
    init_matrix(h_B, K, N, 24);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // ---------- Correctness verification (use a small  N <= 512 is used for full CPU comparison) ----------
    if (N <= 512) {
        run_naive_gemm(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        float* h_C_ref = (float*)malloc(size_C);
        cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);
        verify_result(h_C, h_C_ref, M * N);
        free(h_C_ref);
    } else {
        printf("  [verify] N > 512, skipping full CPU verification (too slow).\n");
    }

    // ---------- Benchmark ----------
    float avg_ms = benchmark_kernel(
    [&]() { run_naive_gemm(d_A, d_B, d_C, M, N, K); });
    
    double gflops = compute_gflops(M, N, K, avg_ms);

    printf("  [bench] avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms, gflops);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}