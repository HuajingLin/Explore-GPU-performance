// v1_shared_tiling.cu
// Progressive Performance Matrix - V1: Shared Memory Tiling
//
// Improvement over V0: Introduces tiling by loading sub-blocks of A and B into shared memory for reuse,
// thereby eliminating a large number of redundant global memory reads.
// Each block is responsible for computing a TILE_SIZE x TILE_SIZE sub-block of C,
// iterating along the K dimension to load tiles and accumulate results. //
// Compile: nvcc -O3 -arch=sm_75 v1_shared_tiling.cu -o v1_shared_tiling
//
// Run: ./v1_shared_tiling [N] [TILE]   (Default N=1024; TILE controlled by macro at compile time)
//
// Verification points (refer to README):
//   - Shared memory usage = 2 * TILE_SIZE * TILE_SIZE * sizeof(float),
//     must be <= 48KB/block; for TILE=32, it is 2*32*32*4 = 8192B, well below the limit.
//   - Recommended to compile and run for TILE_SIZE = 8, 16, and 32 to perform a sensitivity scan.

#include "common.cuh"

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// ---------- Shared Memory Tiled GEMM Kernel ----------
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N,
                                int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;  // Row of C
    int col = blockIdx.x * TILE_SIZE + tx;  // Column of C

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Cooperative loading of A and B sub-tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();  // Ensure the entire tile is loaded

        // Accumulate along the K dimension, reusing data in shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // Ensure computation for this round is complete before overwriting shared memory
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_tiled_gemm(const float* d_A, const float* d_B, float* d_C, int M,
                    int N, int K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = N, K = N;

    printf("=== V1 Shared Memory Tiling GEMM (TILE_SIZE=%d) ===\n", TILE_SIZE);
    printf("M=%d N=%d K=%d\n", M, N, K);

    size_t smem_bytes = 2 * TILE_SIZE * TILE_SIZE * sizeof(float); 
    printf(" shared memory per block = %zu bytes (limit 49152B)\n", smem_bytes); 
    if (smem_bytes > 49152) { 
        fprintf(stderr, "ERROR: shared memory exceeds the limit, please reduce TILE_SIZE\n"); 
        return EXIT_FAILURE; 
    } 

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

    // ---------- Correctness check ---------- 
    if (N <= 512) { 
        run_tiled_gemm(d_A, d_B, d_C, M, N, K); CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        float* h_C_ref = (float*)malloc(size_C);
        cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);
        verify_result(h_C, h_C_ref, M * N);
        free(h_C_ref);
    } else {
        printf("  [verify] N > 512, skipping full CPU verification (too slow); suggest verifying correctness separately using a smaller scale.\n");
    }

    // ---------- Benchmark ----------
    float avg_ms = benchmark_kernel(
    [&]() { run_tiled_gemm(d_A, d_B, d_C, M, N, K); });
    double gflops = compute_gflops(M, N, K, avg_ms);

    printf("  [bench] avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms, gflops);
    printf("  Note: Compare this GFLOPS value with V0; the target is >= 4x V0.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}