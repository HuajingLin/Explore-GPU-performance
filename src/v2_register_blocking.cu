// v2_register_blocking.cu
//
// Compilation: nvcc -O3 -arch=sm_75 v2_register_blocking.cu -o v2_reg4x4
//              (TM/TN can be overridden via -DTM=8 -DTN=8 for experiments comparing register pressure vs. parallelism)
//
// Execution: ./v2_reg4x4 [N]   (Default N=1024; requires M=N=K=N to be divisible by BM/BN/BK)
//

#include "common.cuh"

// ---------- Block Tile / Thread Tile Parameters (can be overridden by compile-time macros) ----------
#ifndef BM
#define BM 64  // Number of output rows handled by each block
#endif
#ifndef BN
#define BN 64  // Number of output columns handled by each block
#endif
#ifndef BK
#define BK 8   // Tile depth loaded per iteration along the K dimension
#endif
#ifndef TM
#define TM 4   // Number of output rows handled by each thread
#endif
#ifndef TN
#define TN 4   // Number of output columns handled by each thread
#endif

// Threads per block = Total output elements per block / Elements handled per thread
#define NUM_THREADS ((BM * BN) / (TM * TN))

//NCU measurements show a wavefront conflict rate of approximately 50%.
//Pad the row width of As from BK to BK+1 so that the address interval across groups 
//becomes TM*(BK+1)—no longer an integer multiple of 32—thereby eliminating this source of conflict.
#define BK_PAD (BK + 1)
// ---------- V2 Register Blocking Kernel ----------
// C[M,N] = A[M,K] * B[K,N]
// grid: (N/BN, M/BM)  block: (NUM_THREADS) 1D layout
__global__ void register_blocking_gemm_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                                    float* __restrict__ C, 
                                                    int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Map 1D threadIdx.x to a 2D thread grid within the block: (BN/TN) x (BM/TM)
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    //__shared__ float As[BM * BK];
    __shared__ float As[BM * BK_PAD];
    __shared__ float Bs[BK * BN];

    // Shift A/B/C pointers to the start of the sub-block assigned to this block
    A += cRow * BM * K; //memory is linear
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    //A is divided into smaller tiles according to 32×8=256 threads. 
    //B is divided into smaller tiles according to 4×64=256 threads.
    // Index for cooperative loading: map NUM_THREADS threads to BM x BK / BK x BN loading tasks
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = NUM_THREADS / BK;

    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = NUM_THREADS / BN;

    // Output accumulators for each thread (in registers): TM * TN elements
    float threadResults[TM * TN] = {0.0f};
    // Temporary registers for each iteration of the K-loop
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- Cooperatively load BM x BK block of A into shared memory ----
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            //As[(innerRowA + loadOffset) * BK + innerColA] =
            As[(innerRowA + loadOffset) * BK_PAD + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
        }

        // ---- Cooperative loading of BK x BN block of B into shared memory ----
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;      // Advance A along the K dimension
        B += BK * N;  // Advance B along the K dimension (across rows)

        // ---- Outer-product accumulation: key optimization point ----
        // In each dotIdx iteration: read TM + TN values ​​from shared memory into registers,
        // yielding TM * TN FMA operations; the memory-access-to-computation ratio is drastically reduced.
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Read the TM rows assigned to this thread — threads within the same warp
            // sharing the same threadRow group read identical addresses; this is a broadcast access
            // and does not cause bank conflicts.
            for (int i = 0; i < TM; ++i) {
                //regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
                regM[i] = As[(threadRow * TM + i) * BK_PAD + dotIdx];
            }

            // Read the TN columns assigned to this thread — similarly, this is a broadcast access.
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            // Perform outer-product accumulation in registers; no further shared memory access.
            #pragma unroll
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) { 
                    threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // ---- Write results from registers back to global memory ----
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int row = cRow * BM + threadRow * TM + resIdxM;
            int col = cCol * BN + threadCol * TN + resIdxN;
            if (row < M && col < N) {  // Note: Only triggered if N is not divisible by BM/BN
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                threadResults[resIdxM * TN + resIdxN];
            }
        }
    }
}

void run_register_blocking_gemm(const float* d_A, 
                                const float* d_B,
                                      float* d_C, 
                                int M, int N, int K) {
    dim3 block(NUM_THREADS);    // (64*64) / (4*4) = 16*16=256, each thread computes 4*4=16 output elements
    dim3 grid(N / BN, M / BM);  // matrix size / block size = number of blocks
    register_blocking_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = N, K = N;

    printf("=== V2 Register Blocking GEMM (BM=%d BN=%d BK=%d TM=%d TN=%d) ===\n",
                                              BM,   BN,   BK,   TM,   TN);
    printf("M=%d N=%d K=%d\n", M, N, K);
    printf("threads/block=%d, each thread computes %d x %d = %d output elements\n",
                          NUM_THREADS,             TM,  TN,  TM * TN);

    // ---- Pre-check: This implementation requires dimensions to be divisible by the block tile size ----
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr,
            "ERROR: M/N/K must be divisible by BM=%d/BN=%d/BK=%d (current implementation "
            "does not perform full padding beyond boundary handling)\n",
            BM, BN, BK);
        return EXIT_FAILURE;
    }
    if (NUM_THREADS % BK != 0 || NUM_THREADS % BN != 0) {
        fprintf(stderr,
            "ERROR: NUM_THREADS(%d) must be divisible by both BK(%d) and BN(%d), "
            "otherwise the stride calculation for cooperative loading will be invalid; please adjust the BM/BN/BK/TM/TN combination\n",
            NUM_THREADS, BK, BN);
        return EXIT_FAILURE;
    }

    //size_t smem_bytes = (BM * BK + BK * BN) * sizeof(float);
    size_t smem_bytes = (BM * BK_PAD + BK * BN) * sizeof(float);
    printf("  shared memory per block = %zu bytes (limit 49152B)\n", smem_bytes);

    if (smem_bytes > 49152) {
        fprintf(stderr, "ERROR: shared memory exceeds the limit\n");
        return EXIT_FAILURE;
    }

    size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

    //matrices in host memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    init_matrix(h_A, M, K, 42);
    init_matrix(h_B, K, N, 24);

    //matrices in device memory(GPU)
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // ---------- Correctness Verification ----------
    if (N <= 512) {
        run_register_blocking_gemm(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

        float* h_C_ref = (float*)malloc(size_C);
        cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);
        verify_result(h_C, h_C_ref, M * N);
        free(h_C_ref);
    } else {
        printf("  [verify] N > 512; skipping full CPU comparison (too slow). It is recommended to verify correctness using a smaller scale separately.\n");
    }

    // ---------- Benchmark ----------
    float avg_ms = benchmark_kernel(
        [&]() { run_register_blocking_gemm(d_A, d_B, d_C, M, N, K); }
    );
    double gflops = compute_gflops(M, N, K, avg_ms);

    printf("  [bench] avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms, gflops);
    printf("  Note: Compare this GFLOPS value with V1; the target is >= 2x V1.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}