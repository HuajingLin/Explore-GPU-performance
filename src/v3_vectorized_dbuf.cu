#include "common.cuh"

#ifndef BM
    #define BM 128
#endif
#ifndef BN
    #define BN 128
#endif
#ifndef BK
    #define BK 8
#endif
#ifndef TM
    #define TM 8
#endif
#ifndef TN
    #define TN 8
#endif

#define NUM_THREADS ((BM * BN) / (TM * TN))

// V3.1
#define AS_PITCH (BM + 4)

// V3.1
__device__ __forceinline__ int bphys4(int n4) {
    return (n4 & 1) * (BN / 8) + (n4 >> 1);
}

// V3.2
__device__ __forceinline__ int bswz_write_col(int tid) {
    int q = (tid & 31) >> 3;
    int r = tid & 7;
    return 2 * (r + 8 * (q & 1)) + (q >> 1);
}

__global__ void vectorized_dbuf_gemm_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                                  float* __restrict__ C, 
                                                  int M, int N, int K) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // Double-buffered shared memory: alternating use of [0] and [1]
    __shared__ float As[2][BK * AS_PITCH];
    __shared__ float Bs[2][BK * BN];

    // move pointers to the block that map to the thread.
    const float* Ab = A + (size_t)cRow * BM * K;
    const float* Bb = B + cCol * BN;
    float* Cb = C + (size_t)cRow * BM * N + cCol * BN;

    // each thread maps to 4 floats, 128 * 2 = 256.
    const int innerRowA = threadIdx.x / (BK / 4);  // 0..BM-1
    const int innerColA = threadIdx.x % (BK / 4);  // 0..(BK/4-1)
    // for B, 8 * 32 = 256
    const int innerRowB = threadIdx.x / (BN / 4);  // 0..BK-1
    //const int innerColB = threadIdx.x % (BN / 4);  // 0..(BN/4-1)
    const int innerColB = tbswz_write_col(threadIdx.x);

    float threadResults[TM * TN] = {0.0f};  //one tile's reasult or one thread's result
    float regM[TM] = {0.0f};    //M loop for calculate one row of tile
    float regN[TN] = {0.0f};
    float4 prefetchA, prefetchB;  // Prefetch registers for double buffering

    // ---------- Prologue: Synchronously load the first 4 floats into buffer 0 ----------
    {
        // each time load 4 floats, 256 * 4 = 1024 floats, which is the size of (BM * BK) = (128 * 8).
        // full load shared memory, the thread maxtrix for A is (128 * 2)
    float4 tmpA = reinterpret_cast<const float4*>( &Ab[innerRowA * K + 0 + innerColA * 4] )[0];
    //A transpose
    As[0][(innerColA * 4 + 0) * AS_PITCH + innerRowA] = tmpA.x;
    As[0][(innerColA * 4 + 1) * AS_PITCH + innerRowA] = tmpA.y;
    As[0][(innerColA * 4 + 2) * AS_PITCH + innerRowA] = tmpA.z;
    As[0][(innerColA * 4 + 3) * AS_PITCH + innerRowA] = tmpA.w;

    //the thread maxtrix for B is (8 * 32)
    float4 tmpB = reinterpret_cast<const float4*>( &Bb[(0 + innerRowB) * N + innerColB * 4] )[0];

    //reinterpret_cast<float4*>(&Bs[0][innerRowB * BN + innerColB * 4])[0] = tmpB;
    reinterpret_cast<float4*>(&Bs[0][innerRowB * BN + bphys4(innerColB) * 4])[0] = tmpB;
    }
    __syncthreads();

    int curBuf = 0;
    //forward on the K dimension in steps of BK
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        int nextBkIdx = bkIdx + BK;
        bool hasNext = nextBkIdx < K;

        // Initiate the global load for the next tile early (store results in registers first,
        // without immediately writing to shared memory) to overlap the load latency
        // with the subsequent computation instructions.
        if (hasNext) {
            prefetchA = reinterpret_cast<const float4*>(  &Ab[innerRowA * K + nextBkIdx + innerColA * 4]  )[0];
            prefetchB = reinterpret_cast<const float4*>(  &Bb[(nextBkIdx + innerRowB) * N + innerColB * 4])[0];
        }

        // ---- Compute using the current buffer (data was synchronized in the previous round, so it is safe to read) ----
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Vectorized read for regM: TM=8; use two float4 reads instead of eight scalar reads
            #pragma unroll
            for (int i = 0; i < TM; i += 4) {
                float4 tmp = reinterpret_cast<float4*>(
                    &As[curBuf][dotIdx * AS_PITCH + threadRow * TM + i])[0];
                regM[i + 0] = tmp.x;
                regM[i + 1] = tmp.y;
                regM[i + 2] = tmp.z;
                regM[i + 3] = tmp.w;
            }
            // Vectorized read for regN: TN=8; similarly, use two float4 reads
            #pragma unroll
            for (int i = 0; i < TN; i += 4) {
                float4 tmp = reinterpret_cast<float4*>(
                    &Bs[curBuf][dotIdx * BN +
                                bphys4(threadCol * 2 + i / 4) * 4])[0];
                regN[i + 0] = tmp.x;
                regN[i + 1] = tmp.y;
                regN[i + 2] = tmp.z;
                regN[i + 3] = tmp.w;
            }
            #pragma unroll
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                #pragma unroll
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN]; }
            }
        }

        // ---- Computation complete; write the previously prefetched data for the next tile into the other buffer ----
        if (hasNext) {
            int nextBuf = 1 - curBuf;
            As[nextBuf][(innerColA * 4 + 0) * AS_PITCH + innerRowA] = prefetchA.x;
            As[nextBuf][(innerColA * 4 + 1) * AS_PITCH + innerRowA] = prefetchA.y;
            As[nextBuf][(innerColA * 4 + 2) * AS_PITCH + innerRowA] = prefetchA.z;
            As[nextBuf][(innerColA * 4 + 3) * AS_PITCH + innerRowA] = prefetchA.w;
            //reinterpret_cast<float4*>( &Bs[nextBuf][innerRowB * BN + innerColB * 4] )[0] = prefetchB;
            reinterpret_cast<float4*>(
                &Bs[nextBuf][innerRowB * BN + bphys4(innerColB) * 4])[0] =
                prefetchB;
            __syncthreads();  // Ensure all threads have finished writing to nextBuf before safe reading in the next iteration
            curBuf = nextBuf;
        }
    }

    // ---- Vectorized write-back of results: TN=8; write out using two float4 operations per row ----
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (int resIdxN = 0; resIdxN< TN; resIdxN += 4) { 
            float4 out; 
            out.x = threadResults[resIdxM * TN + resIdxN + 0]; 
            out.y = threadResults[resIdxM * TN + resIdxN + 1]; 
            out.z = threadResults[resIdxM * TN + resIdxN + 2]; 
            out.w = threadResults[resIdxM * TN + resIdxN + 3]; 
            reinterpret_cast<float4*>( &Cb[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = out; 
        } 
    }
}

void run_vectorized_dbuf_gemm(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) { 
    dim3 block(NUM_THREADS); 
    dim3 grid(N / BN, M / BM); 
    vectorized_dbuf_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

int main(int argc, char** argv) { 
    int N = (argc > 1) ? atoi(argv[1]) : 1024; 
    int M = N, K = N; 

    printf( "=== V3 Vectorized + Double-Buffered GEMM (BM=%d BN=%d BK=%d TM=%d TN=%d) ===\n", BM, BN, BK, TM, TN); 
    printf("M=%d N=%d K=%d\n", M, N, K); 
    printf("threads/block=%d, each thread calculates %d x %d = %d output elements\n", NUM_THREADS, TM, TN, TM * TN); 

    // ---- Pre-check ---- 
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) { 
        fprintf(stderr, "ERROR: M/N/K must be divisible by BM=%d/BN=%d/BK=%d\n", BM, BN, BK);
        return EXIT_FAILURE;
    }
    if (BK % 4 != 0 || BN % 4 != 0 || TM % 4 != 0 || TN % 4 != 0) {
        fprintf(stderr, "ERROR: BK/BN/TM/TN must all be multiples of 4 (required for float4 vectorization)\n");
        return EXIT_FAILURE;
    }
    if (NUM_THREADS != BM * (BK / 4) || NUM_THREADS != BK * (BN / 4)) {
        fprintf(stderr,
        "ERROR: This simplified version requires NUM_THREADS(%d) == BM*(BK/4)(%d) == "
        "BK*(BN/4)(%d) to hold simultaneously; otherwise, the thread-to-data mapping "
        "for vectorized loads is incomplete (stride loops are not implemented; "
        "adjust the BM/BN/BK/TM/TN combination)\n",
        NUM_THREADS, BM * (BK / 4), BK * (BN / 4));
        return EXIT_FAILURE;
    }

    size_t smem_bytes = 2 * (BK * AS_PITCH + BK * BN) * sizeof(float);  // Double buffering (x2)
    printf("  shared memory per block = %zu bytes (limit 49152B, including double buffering x2)\n", smem_bytes);
    if (smem_bytes > 49152) {
        fprintf(stderr, "ERROR: Shared memory exceeds limit\n");
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

    // ---------- Correctness check: This kernel index logic is complex, be sure to confirm PASS first ---------- 
    if (N <= 512) { 
        run_vectorized_dbuf_gemm(d_A, d_B, d_C, M, N, K); 
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost)); 

        float* h_C_ref = (float*)malloc(size_C); 
        cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K); 
        verify_result(h_C, h_C_ref, M * N); 
        free(h_C_ref); 
    } else { 
        printf( " [verify first with N<=512]\n");
    }

    // ---------- Benchmark ----------
    float avg_ms = benchmark_kernel( [&]() { run_vectorized_dbuf_gemm(d_A, d_B, d_C, M, N, K); } );
    double gflops = compute_gflops(M, N, K, avg_ms);

    printf("  [bench] avg_time=%.3f ms, GFLOPS=%.2f\n", avg_ms, gflops);
    printf("  Note: Compare against V2 (TM=TN=8, 1974 GFLOPS @ N=2048)\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}