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
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, \
                    __LINE__, (int)status);                                \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// ---------- FP32matrix to FP16 (host)----------
void convert_to_half(const float* src, half* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = __float2half(src[i]);
    }
}

// ---------- V4 base WMMA Tensor Core Kernel ----------
__global__ void wmma_naive_gemm_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float* __restrict__ C, int M, int N,int K) {
    //each warp:16x16
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
            // load_matrix_sync read WMMA_M(or K) x WMMA_K(or N)
            wmma::load_matrix_sync(a_frag, A + (size_t)aRow * K + k, K);
            wmma::load_matrix_sync(b_frag, B + (size_t)k * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        wmma::store_matrix_sync(C + (size_t)aRow * N + bCol, acc_frag, N,
                                 wmma::mem_row_major);
    }
}

void run_wmma_gemm(const half* d_A, const half* d_B, float* d_C, int M,
                    int N, int K) {
    dim3 blockDim(128, 4);
    dim3 gridDim((M + (WMMA_M * (blockDim.x / 32)) - 1) /
                     (WMMA_M * (blockDim.x / 32)),
                 (N + (WMMA_N * blockDim.y) - 1) / (WMMA_N * blockDim.y));
    wmma_naive_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ---------- V4b: Shared Memory Tiling & WMMA Kernel ----------

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16  

__global__ void wmma_tiled_gemm_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float* __restrict__ C, int M, int N,
                                        int K) {
    int cRow = blockIdx.x;  
    int cCol = blockIdx.y;  

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..511
    int totalThreads = blockDim.x * blockDim.y;         // 512

    int warpM_local = threadIdx.x / warpSize;  // 0..3
    int warpN_local = threadIdx.y;             // 0..3

    int aRowBase = cRow * TILE_M + warpM_local * WMMA_M;
    int bColBase = cCol * TILE_N + warpN_local * WMMA_N;

    __shared__ half As[TILE_M * TILE_K];  
    __shared__ half Bs[TILE_K * TILE_N];  

    const half* Ab = A + (size_t)cRow * TILE_M * K;
    const half* Bb = B + cCol * TILE_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {
        // ---- block load As (TILE_M x TILE_K) ----
        for (int idx = tid; idx < TILE_M * TILE_K; idx += totalThreads) {
            int localRow = idx / TILE_K;
            int localCol = idx % TILE_K;
            As[idx] = Ab[(size_t)localRow * K + bkIdx + localCol];
        }
        // ---- load Bs (TILE_K x TILE_N) ----
        for (int idx = tid; idx < TILE_K * TILE_N; idx += totalThreads) {
            int localRow = idx / TILE_N;
            int localCol = idx % TILE_N;
            Bs[idx] = Bb[(size_t)(bkIdx + localRow) * N + localCol];
        }
        __syncthreads(); 

        if (aRowBase < M && bColBase < N) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
                wmma::load_matrix_sync(
                    a_frag, As + warpM_local * WMMA_M * TILE_K + kk, TILE_K);
                wmma::load_matrix_sync(
                    b_frag, Bs + kk * TILE_N + warpN_local * WMMA_N, TILE_N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        __syncthreads();
    }

    if (aRowBase < M && bColBase < N) {
        wmma::store_matrix_sync(C + (size_t)aRowBase * N + bColBase, acc_frag,
                                 N, wmma::mem_row_major);
    }
}

void run_wmma_tiled_gemm(const half* d_A, const half* d_B, float* d_C, int M,
                          int N, int K) {
    dim3 blockDim(128, 4);
    dim3 gridDim(M / TILE_M, N / TILE_N);
    wmma_tiled_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ---------- V4d: Tiled WMMA + Double Buffering ----------

#define THREADS_PER_BLOCK_TILED 512  // blockDim(128,4)=128*4=512
#define ELEMS_PER_THREAD_A ((TILE_M * TILE_K) / THREADS_PER_BLOCK_TILED)
#define ELEMS_PER_THREAD_B ((TILE_K * TILE_N) / THREADS_PER_BLOCK_TILED)

__global__ void wmma_tiled_dbuf_gemm_kernel(const half* __restrict__ A,
                                             const half* __restrict__ B,
                                             float* __restrict__ C, int M,
                                             int N, int K) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;  // 512

    int warpM_local = threadIdx.x / warpSize;
    int warpN_local = threadIdx.y;

    int aRowBase = cRow * TILE_M + warpM_local * WMMA_M;
    int bColBase = cCol * TILE_N + warpN_local * WMMA_N;

    __shared__ half As[2][TILE_M * TILE_K];  // 双缓冲
    __shared__ half Bs[2][TILE_K * TILE_N];

    const half* Ab = A + (size_t)cRow * TILE_M * K;
    const half* Bb = B + cCol * TILE_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    half prefetchA[ELEMS_PER_THREAD_A];
    half prefetchB[ELEMS_PER_THREAD_B];

    // ---- Prologue: load no.0 tile to buffer 0 ----
    for (int idx = tid; idx < TILE_M * TILE_K; idx += totalThreads) {
        int localRow = idx / TILE_K;
        int localCol = idx % TILE_K;
        As[0][idx] = Ab[(size_t)localRow * K + 0 + localCol];
    }
    for (int idx = tid; idx < TILE_K * TILE_N; idx += totalThreads) {
        int localRow = idx / TILE_N;
        int localCol = idx % TILE_N;
        Bs[0][idx] = Bb[(size_t)(0 + localRow) * N + localCol];
    }
    __syncthreads();

    int curBuf = 0;

    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {
        int nextBkIdx = bkIdx + TILE_K;
        bool hasNext = nextBkIdx < K;

        if (hasNext) {
            int i = 0;
            for (int idx = tid; idx < TILE_M * TILE_K;
                 idx += totalThreads, ++i) {
                int localRow = idx / TILE_K;
                int localCol = idx % TILE_K;
                prefetchA[i] = Ab[(size_t)localRow * K + nextBkIdx + localCol];
            }
            i = 0;
            for (int idx = tid; idx < TILE_K * TILE_N;
                 idx += totalThreads, ++i) {
                int localRow = idx / TILE_N;
                int localCol = idx % TILE_N;
                prefetchB[i] =
                    Bb[(size_t)(nextBkIdx + localRow) * N + localCol];
            }
        }

        if (aRowBase < M && bColBase < N) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
                wmma::load_matrix_sync(a_frag,
                                        As[curBuf] +
                                            warpM_local * WMMA_M * TILE_K + kk,
                                        TILE_K);
                wmma::load_matrix_sync(
                    b_frag, Bs[curBuf] + kk * TILE_N + warpN_local * WMMA_N,
                    TILE_N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        if (hasNext) {
            int nextBuf = 1 - curBuf;
            int i = 0;
            for (int idx = tid; idx < TILE_M * TILE_K;
                 idx += totalThreads, ++i) {
                As[nextBuf][idx] = prefetchA[i];
            }
            i = 0;
            for (int idx = tid; idx < TILE_K * TILE_N;
                 idx += totalThreads, ++i) {
                Bs[nextBuf][idx] = prefetchB[i];
            }
            __syncthreads();
            curBuf = nextBuf;
        }
    }

    if (aRowBase < M && bColBase < N) {
        wmma::store_matrix_sync(C + (size_t)aRowBase * N + bColBase, acc_frag,
                                 N, wmma::mem_row_major);
    }
}

void run_wmma_tiled_dbuf_gemm(const half* d_A, const half* d_B, float* d_C,
                               int M, int N, int K) {
    dim3 blockDim(128, 4);
    dim3 gridDim(M / TILE_M, N / TILE_N);
    wmma_tiled_dbuf_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

#define WARP_TILE_M 4  // number of WMMA_M tiles each warp computes, M direction
#define WARP_TILE_N 2  // number of WMMA_N tiles each warp computes, N direction
#define WARPS_M (TILE_M / (WARP_TILE_M * WMMA_M))  // = 64/32 = 2 warps along M
#define WARPS_N (TILE_N / (WARP_TILE_N * WMMA_N))  // = 64/32 = 2 warps along N

__global__ void wmma_warptile_gemm_kernel(const half* __restrict__ A,
                                           const half* __restrict__ B,
                                           float* __restrict__ C, int M,
                                           int N, int K) {
    int cRow = blockIdx.x;  // block's M-direction tile index
    int cCol = blockIdx.y;  // block's N-direction tile index

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // flattened thread id within block
    int totalThreads = blockDim.x * blockDim.y;         // 128 for this kernel

    int warpM_local = threadIdx.x / warpSize;  // 0..WARPS_M-1 (0..1)
    int warpN_local = threadIdx.y;             // 0..WARPS_N-1 (0..1)

    // This warp's output region, in local (block-relative) coordinates
    int warpRowBase = warpM_local * WARP_TILE_M * WMMA_M;  // 0 or 32
    int warpColBase = warpN_local * WARP_TILE_N * WMMA_N;  // 0 or 32

    // This warp's output region, in global matrix coordinates
    int aRowBase = cRow * TILE_M + warpRowBase;
    int bColBase = cCol * TILE_N + warpColBase;

    __shared__ half As[TILE_M * TILE_K];  // row-major, row stride = TILE_K
    __shared__ half Bs[TILE_K * TILE_N];  // row-major, row stride = TILE_N

    const half* Ab = A + (size_t)cRow * TILE_M * K;
    const half* Bb = B + cCol * TILE_N;

    // Each warp holds WARP_TILE_M a_frags, WARP_TILE_N b_frags, and the
    // full WARP_TILE_M x WARP_TILE_N cross-product of acc_frags
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                    wmma::row_major>
        b_frag[WARP_TILE_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_K) {
        // ---- cooperative load of As (TILE_M x TILE_K) ----
        for (int idx = tid; idx < TILE_M * TILE_K; idx += totalThreads) {
            int localRow = idx / TILE_K;
            int localCol = idx % TILE_K;
            As[idx] = Ab[(size_t)localRow * K + bkIdx + localCol];
        }
        // ---- cooperative load of Bs (TILE_K x TILE_N) ----
        for (int idx = tid; idx < TILE_K * TILE_N; idx += totalThreads) {
            int localRow = idx / TILE_N;
            int localCol = idx % TILE_N;
            Bs[idx] = Bb[(size_t)(bkIdx + localRow) * N + localCol];
        }
        __syncthreads();  // wait until shared memory tile is fully loaded

        // ---- each warp loads its WARP_TILE_M a_frags and WARP_TILE_N
        // b_frags once, then reuses them across the WARP_TILE_M x
        // WARP_TILE_N cross-product of mma_sync calls (this is exactly
        // the "1:1 load:compute ratio" described above) ----
        if (aRowBase < M && bColBase < N) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
#pragma unroll
                for (int i = 0; i < WARP_TILE_M; ++i) {
                    wmma::load_matrix_sync(
                        a_frag[i],
                        As + (warpRowBase + i * WMMA_M) * TILE_K + kk,
                        TILE_K);
                }
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::load_matrix_sync(
                        b_frag[j],
                        Bs + kk * TILE_N + (warpColBase + j * WMMA_N),
                        TILE_N);
                }
#pragma unroll
                for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                    for (int j = 0; j < WARP_TILE_N; ++j) {
                        wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j],
                                        acc_frag[i][j]);
                    }
                }
            }
        }

        __syncthreads();  // make sure every warp is done reading before the
                           // next iteration overwrites shared memory
    }

    // ---- write back all WARP_TILE_M x WARP_TILE_N accumulator tiles ----
    if (aRowBase < M && bColBase < N) {
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int row = aRowBase + i * WMMA_M;
                int col = bColBase + j * WMMA_N;
                wmma::store_matrix_sync(C + (size_t)row * N + col,
                                         acc_frag[i][j], N,
                                         wmma::mem_row_major);
            }
        }
    }
}

void run_wmma_warptile_gemm(const half* d_A, const half* d_B, float* d_C,
                             int M, int N, int K) {
    // blockDim: WARPS_M*32 threads along x (2 warps x 32 = 64), WARPS_N
    // along y (2) -> 128 threads = 4 warps per block, covering the same
    // TILE_M x TILE_N = 64x64 output region as V4b/c/d, just with fewer,
    // "fatter" warps instead of 16 "thin" ones.
    dim3 blockDim(WARPS_M * 32, WARPS_N);
    dim3 gridDim(M / TILE_M, N / TILE_N);
    wmma_warptile_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ---------- cuBLAS

void run_cublas_gemm(cublasHandle_t handle, const half* d_A, const half* d_B,
                      float* d_C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = N, K = N;

    printf("=== V4 WMMA Tensor Core GEMM (FP16 input, FP32 accumulate) ===\n");
    printf("M=%d N=%d K=%d\n", M, N, K);

    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        fprintf(stderr, "ERROR: M/N/K Must be divisible by 16.\n");
        return EXIT_FAILURE;
    }
    if (M % TILE_M != 0 || N % TILE_N != 0 || K % TILE_K != 0) {
        fprintf(stderr,
                "ERROR: M/N Must be divisible by %d, K Must be divisible by TILE_K=%d\n",
                TILE_M, TILE_K);
        return EXIT_FAILURE;
    }

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    // ---- host: FP32 to FP16 ----
    float *h_A_f32, *h_B_f32, *h_C_wmma, *h_C_tiled, *h_C_dbuf, *h_C_warptile, *h_C_cublas;
    h_A_f32 = (float*)malloc(size_A * sizeof(float));
    h_B_f32 = (float*)malloc(size_B * sizeof(float));
    h_C_wmma = (float*)malloc(size_C * sizeof(float));
    h_C_tiled = (float*)malloc(size_C * sizeof(float));
    h_C_dbuf = (float*)malloc(size_C * sizeof(float));
    h_C_warptile = (float*)malloc(size_C * sizeof(float));
    h_C_cublas = (float*)malloc(size_C * sizeof(float));

    init_matrix(h_A_f32, M, K, 42);
    init_matrix(h_B_f32, K, N, 24);

    half *h_A_f16, *h_B_f16;
    h_A_f16 = (half*)malloc(size_A * sizeof(half));
    h_B_f16 = (half*)malloc(size_B * sizeof(half));
    convert_to_half(h_A_f32, h_A_f16, size_A);
    convert_to_half(h_B_f32, h_B_f16, size_B);

    // ---- device memory ----
    half *d_A, *d_B;
    float *d_C_wmma, *d_C_tiled, *d_C_dbuf, *d_C_warptile, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_dbuf, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_warptile, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_f16, size_A * sizeof(half),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_f16, size_B * sizeof(half),
                           cudaMemcpyHostToDevice));

    // ---------- verify ----------
    if (N <= 512) {
        run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_wmma, d_C_wmma, size_C * sizeof(float),
                               cudaMemcpyDeviceToHost));

        float* h_C_ref = (float*)malloc(size_C * sizeof(float));
        cpu_gemm_ref(h_A_f32, h_B_f32, h_C_ref, M, N, K);

        printf("  [verify naive wmma kernel] (FP16 atol=0.5/rtol=5%%)\n");
        verify_result(h_C_wmma, h_C_ref, (int)size_C, 0.5f, 0.05f);

        run_wmma_tiled_gemm(d_A, d_B, d_C_tiled, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C_tiled, size_C * sizeof(float),
                               cudaMemcpyDeviceToHost));
        printf("  [verify tiled wmma kernel]\n");
        verify_result(h_C_tiled, h_C_ref, (int)size_C, 0.5f, 0.05f);

        run_wmma_tiled_dbuf_gemm(d_A, d_B, d_C_dbuf, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_dbuf, d_C_dbuf, size_C * sizeof(float),
                               cudaMemcpyDeviceToHost));
        printf("  [verify tiled+dbuf wmma kernel]\n");
        verify_result(h_C_dbuf, h_C_ref, (int)size_C, 0.5f, 0.05f);

        run_wmma_warptile_gemm(d_A, d_B, d_C_warptile, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C_warptile, d_C_warptile,
                               size_C * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  [verify warptile wmma kernel]\n");
        verify_result(h_C_warptile, h_C_ref, (int)size_C, 0.5f, 0.05f);

        free(h_C_ref);
    } else {
        printf("  [verify] N>512 skip\n");
    }

    // ---------- Benchmark: naive wmma kernel ----------
    float avg_ms_wmma =
        benchmark_kernel([&]() { run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K); });
    double gflops_wmma = compute_gflops(M, N, K, avg_ms_wmma);
    printf("  [bench] naive WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n",
           avg_ms_wmma, gflops_wmma);

    // ---------- Benchmark: shared memory tiled wmma kernel ----------
    float avg_ms_tiled = benchmark_kernel(
        [&]() { run_wmma_tiled_gemm(d_A, d_B, d_C_tiled, M, N, K); });
    double gflops_tiled = compute_gflops(M, N, K, avg_ms_tiled);
    printf("  [bench] tiled WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n",
           avg_ms_tiled, gflops_tiled);
    printf("  [compare] tiled over naive promote: %.2fx\n",
           gflops_tiled / gflops_wmma);

    // ---------- Benchmark: tiled + double buffering wmma kernel ----------
    float avg_ms_dbuf = benchmark_kernel(
        [&]() { run_wmma_tiled_dbuf_gemm(d_A, d_B, d_C_dbuf, M, N, K); });
    double gflops_dbuf = compute_gflops(M, N, K, avg_ms_dbuf);
    printf("  [bench] tiled+dbuf WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n",
           avg_ms_dbuf, gflops_dbuf);
    printf("  [compare] tiled+dbuf over tiled promote: %.2fx\n",
           gflops_dbuf / gflops_tiled);

    // ---------- Benchmark: warp-level multi-fragment tiling wmma kernel ----------
    float avg_ms_warptile = benchmark_kernel(
        [&]() { run_wmma_warptile_gemm(d_A, d_B, d_C_warptile, M, N, K); });
    double gflops_warptile = compute_gflops(M, N, K, avg_ms_warptile);
    printf("  [bench] warptile WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n",
           avg_ms_warptile, gflops_warptile);
    printf("  [compare] warptile over tiled promote: %.2fx\n",
           gflops_warptile / gflops_tiled);

    // ---------- Benchmark: cuBLAS run Tensor Core----------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float avg_ms_cublas = benchmark_kernel([&]() {
        run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    });
    double gflops_cublas = compute_gflops(M, N, K, avg_ms_cublas);
    printf("  [bench] cuBLAS(TensorCore): avg_time=%.3f ms, GFLOPS=%.2f\n",
           avg_ms_cublas, gflops_cublas);

    double pct_naive = 100.0 * gflops_wmma / gflops_cublas;
    double pct_tiled = 100.0 * gflops_tiled / gflops_cublas;
    double pct_dbuf = 100.0 * gflops_dbuf / gflops_cublas;
    double pct_warptile = 100.0 * gflops_warptile / gflops_cublas;
    printf("  [compare] naive kernel achieving cuBLAS performance %.1f%%\n", pct_naive);
    printf("  [compare] tiled kernel achieving cuBLAS performance %.1f%%\n", pct_tiled);
    printf("  [compare] tiled+dbuf kernel achieving cuBLAS performance %.1f%%\n", pct_dbuf);
    printf("  [compare] warptile kernel achieving cuBLAS performance %.1f%%\n", pct_warptile);
    printf("  (Acceptance Criteria: N>=1024 >= 70%% Deemed to meet the standard.)\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_dbuf));
    CUDA_CHECK(cudaFree(d_C_warptile));
    CUDA_CHECK(cudaFree(d_C_cublas));
    free(h_A_f32);
    free(h_B_f32);
    free(h_A_f16);
    free(h_B_f16);
    free(h_C_wmma);
    free(h_C_tiled);
    free(h_C_dbuf);
    free(h_C_warptile);
    free(h_C_cublas);

    return 0;
}
