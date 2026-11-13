// v4_wmma_tensorcore.cu
//
// Build (Colab T4, sm_75):
//   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
// Run:
//   ./build/v4_wmma 512     # verifies every kernel against the CPU reference
//   ./build/v4_wmma 1024    # benchmark only (verification skipped for N > 512)
//   ./build/v4_wmma 2048
//   ./build/v4_wmma 4096
// Judge "% of cuBLAS" on the [robust] median lines, not the single-shot
// [compare] lines: T4 clocks/power state swing cuBLAS by up to ~40%.
// The [attr] lines report regs / spills / smem / blocks-per-SM per config.

#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <algorithm>
#include <vector>

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

// ---------- V4e: Block tile + warp tile + padded smem + int4 loads + dbuf ----------
//
// Template parameters:
//   kBM x kBN  : output tile per block
//   kBK       : K step per shared-memory stage (multiple of 16)
//   kWM x kWN  : output tile per warp (multiples of 16)
// Requirements: M % kBM == 0, N % kBN == 0, K % kBK == 0.

// kMinBlocks > 1 asks the compiler to fit that many blocks per SM, which caps
// registers per thread (65536 / (threads * kMinBlocks)); watch for spills.
// kStages = 2: two shared-memory buffers, one barrier per K step.
// kStages = 1: one shared-memory buffer (half the smem), two barriers per K
//              step. The next tile is still prefetched into registers, so
//              global-load latency is still overlapped with the MMAs.
template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2>
__global__ void __launch_bounds__((kBM / kWM) * (kBN / kWN) * 32, kMinBlocks)
wmma_blocktile_gemm_kernel(const half* __restrict__ A,
                           const half* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
    constexpr int kWarpsN  = kBN / kWN;
    constexpr int kThreads = (kBM / kWM) * kWarpsN * 32;
    constexpr int kFragM   = kWM / WMMA_M;
    constexpr int kFragN   = kWN / WMMA_N;

    // +8 halves (16 B) of padding per row: ldm stays a multiple of 8 (WMMA
    // requirement for half), and the 8 rows read by each ldmatrix land on
    // 8 different 16-byte bank groups -> no bank conflicts.
    constexpr int kPad   = 8;
    constexpr int kAsLd = kBK + kPad;
    constexpr int kBsLd = kBN + kPad;

    // 16-byte vectors (8 halves)
    constexpr int kAVecsPerRow = kBK / 8;
    constexpr int kBVecsPerRow = kBN / 8;
    constexpr int kAPerT = (kBM * kBK / 8) / kThreads;
    constexpr int kBPerT = (kBK * kBN / 8) / kThreads;
    static_assert((kBM * kBK / 8) % kThreads == 0, "A tile must split evenly");
    static_assert((kBK * kBN / 8) % kThreads == 0, "B tile must split evenly");
    static_assert(kBK % WMMA_K == 0, "kBK must be a multiple of 16");
    static_assert(kWM % WMMA_M == 0 && kWN % WMMA_N == 0, "warp tile vs WMMA");

    // load_matrix_sync needs 256-bit aligned pointers
    static_assert(kStages == 1 || kStages == 2, "kStages must be 1 or 2");
    __shared__ __align__(128) half As[kStages][kBM * kAsLd];
    __shared__ __align__(128) half Bs[kStages][kBK * kBsLd];

    const int tid     = threadIdx.x;
    const int warpId  = tid / 32;
    const int warpRow = warpId / kWarpsN;
    const int warpCol = warpId % kWarpsN;

    // blockIdx.x walks N so neighbouring blocks share the same A rows (L2 reuse)
    const int blockRow = blockIdx.y * kBM;
    const int blockCol = blockIdx.x * kBN;

    const half* Ag = A + (size_t)blockRow * K;
    const half* Bg = B + blockCol;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        a_frag[kFragM];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        b_frag[kFragN];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[kFragM][kFragN];

#pragma unroll
    for (int i = 0; i < kFragM; ++i)
#pragma unroll
        for (int j = 0; j < kFragN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    int4 ra[kAPerT];
    int4 rb[kBPerT];

    auto load_global = [&](int k0) {
#pragma unroll
        for (int i = 0; i < kAPerT; ++i) {
            int idx = tid + i * kThreads;
            int r = idx / kAVecsPerRow;
            int c = (idx % kAVecsPerRow) * 8;
            ra[i] = *reinterpret_cast<const int4*>(Ag + (size_t)r * K + k0 + c);
        }
#pragma unroll
        for (int i = 0; i < kBPerT; ++i) {
            int idx = tid + i * kThreads;
            int r = idx / kBVecsPerRow;
            int c = (idx % kBVecsPerRow) * 8;
            rb[i] = *reinterpret_cast<const int4*>(Bg + (size_t)(k0 + r) * N + c);
        }
    };

    auto store_shared = [&](int buf) {
#pragma unroll
        for (int i = 0; i < kAPerT; ++i) {
            int idx = tid + i * kThreads;
            int r = idx / kAVecsPerRow;
            int c = (idx % kAVecsPerRow) * 8;
            *reinterpret_cast<int4*>(&As[buf][r * kAsLd + c]) = ra[i];
        }
#pragma unroll
        for (int i = 0; i < kBPerT; ++i) {
            int idx = tid + i * kThreads;
            int r = idx / kBVecsPerRow;
            int c = (idx % kBVecsPerRow) * 8;
            *reinterpret_cast<int4*>(&Bs[buf][r * kBsLd + c]) = rb[i];
        }
    };

    // ---- prologue ----
    load_global(0);
    store_shared(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += kBK) {
        const bool hasNext = (k0 + kBK) < K;

        // issue the next tile's global loads first; latency overlaps the MMAs
        if (hasNext) load_global(k0 + kBK);

        const half* As_cur = As[buf];
        const half* Bs_cur = Bs[buf];

#pragma unroll
        for (int kk = 0; kk < kBK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < kFragM; ++i)
                wmma::load_matrix_sync(
                    a_frag[i], As_cur + (warpRow * kWM + i * WMMA_M) * kAsLd + kk,
                    kAsLd);
#pragma unroll
            for (int j = 0; j < kFragN; ++j)
                wmma::load_matrix_sync(
                    b_frag[j], Bs_cur + kk * kBsLd + warpCol * kWN + j * WMMA_N,
                    kBsLd);
#pragma unroll
            for (int i = 0; i < kFragM; ++i)
#pragma unroll
                for (int j = 0; j < kFragN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        if (hasNext) {
            if (kStages == 2) {
                // One barrier is enough: buf^1 was last read in the previous
                // iteration, and every warp has already passed that barrier.
                store_shared(buf ^ 1);
                __syncthreads();
                buf ^= 1;
            } else {
                // Single buffer: wait until every warp is done reading it,
                // overwrite it from the prefetch registers, wait again.
                __syncthreads();
                store_shared(0);
                __syncthreads();
            }
        }
    }

    // ---- epilogue ----
#pragma unroll
    for (int i = 0; i < kFragM; ++i)
#pragma unroll
        for (int j = 0; j < kFragN; ++j) {
            int row = blockRow + warpRow * kWM + i * WMMA_M;
            int col = blockCol + warpCol * kWN + j * WMMA_N;
            wmma::store_matrix_sync(C + (size_t)row * N + col, acc[i][j], N,
                                    wmma::mem_row_major);
        }
}

template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2>
bool v4e_shape_ok(int M, int N, int K) {
    return (M % kBM == 0) && (N % kBN == 0) && (K % kBK == 0);
}

template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2>
void run_wmma_blocktile_gemm(const half* d_A, const half* d_B, float* d_C,
                             int M, int N, int K) {
    constexpr int kThreads = (kBM / kWM) * (kBN / kWN) * 32;
    dim3 block(kThreads);
    dim3 grid(N / kBN, M / kBM);
    wmma_blocktile_gemm_kernel<kBM, kBN, kBK, kWM, kWN, kMinBlocks, kStages>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

// Prints registers, local memory (spills), static smem and how many blocks
// fit per SM, as reported by the runtime for the compiled kernel.
template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2>
void print_v4e_attrs(const char* name) {
    auto kern = wmma_blocktile_gemm_kernel<kBM, kBN, kBK, kWM, kWN, kMinBlocks, kStages>;
    constexpr int kThreads = (kBM / kWM) * (kBN / kWN) * 32;
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kern));
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kern,
                                                             kThreads, 0));
    printf("  [attr] %-18s threads=%3d regs=%3d local(spill)=%4zu B "
           "smem=%6zu B blocks/SM=%d\n",
           name, kThreads, attr.numRegs, attr.localSizeBytes,
           attr.sharedSizeBytes, blocksPerSM);
}

// ---------- V4e configs kept after the T4 sweep ----------
//
// Sweep history (Colab T4, robust median % of cuBLAS; see the write-up):
//   128x128 (8 warps 64x32, 2 smem bufs, 1 block/SM) ......... 69% @1024
//   128x128-k16 / -k16m2 (BK=16, 2 blocks/SM) ................ 59% @1024  (dropped)
//   128x128-w64 (4 warps 64x64, 2 bufs, 1 block/SM) .......... 69% @1024  (dropped)
//   128x128-1stg (8 warps, 1 buf, launch_bounds(256,2), spills) 70% @1024  (dropped)
//   64x64 (4 warps 32x32) ..................................... 54% @1024  (dropped)
// Kept:
//   V4e-128x64   : 4 warps of 64x32, BK=32, 2 smem bufs. Best when the
//                  128x128 grid has fewer blocks than SMs (N=512: 77.7%).
//   V4e-w64-1stg : 128x128 tile, 4 warps of 64x64, BK=32, ONE smem buffer
//                  (~19 KB). 254 regs x 128 threads and 19 KB both allow
//                  2 blocks/SM. Best for N >= 1024 (76-77% @1024,
//                  68-70% @2048, 82% @4096).
//                  254 regs is 1 below the limit: re-check local(spill)
//                  after ANY edit to the kernel.
#define V4E_CFG_128x64 128, 64, 32, 64, 32
#define V4E_CFG_W64_1STG 128, 128, 32, 64, 64, 1, 1
#define V4E_NUM_CFGS 3
static const char* V4E_NAMES[V4E_NUM_CFGS] = {"V4e-128x64", "V4e-w64-1stg",
                                              "V4e-auto"};

void print_all_v4e_attrs() {
    print_v4e_attrs<V4E_CFG_128x64>(V4E_NAMES[0]);
    print_v4e_attrs<V4E_CFG_W64_1STG>(V4E_NAMES[1]);
}

// Size-based dispatcher. The T4 has 40 SMs: if a 128x128 grid would give
// fewer blocks than SMs, use the smaller 128x64 tile so every SM gets work.
// Returns the config index (0 or 1), or -1 if neither shape fits.
int v4e_auto_choice(int M, int N, int K) {
    const int kNumSMs = 40;  // Tesla T4
    bool w64_ok = v4e_shape_ok<V4E_CFG_W64_1STG>(M, N, K);
    bool c64_ok = v4e_shape_ok<V4E_CFG_128x64>(M, N, K);
    if (w64_ok && (M / 128) * (N / 128) >= kNumSMs) return 1;
    if (c64_ok) return 0;
    if (w64_ok) return 1;
    return -1;
}

// Launch config `cfg` (0..V4E_NUM_CFGS-1). Returns false if the shape doesn't fit.
bool run_v4e_cfg(int cfg, const half* d_A, const half* d_B, float* d_C, int M,
                 int N, int K) {
    if (cfg == 2) cfg = v4e_auto_choice(M, N, K);
    switch (cfg) {
        case 0:
            if (!v4e_shape_ok<V4E_CFG_128x64>(M, N, K)) return false;
            run_wmma_blocktile_gemm<V4E_CFG_128x64>(d_A, d_B, d_C, M, N, K);
            return true;
        case 1:
            if (!v4e_shape_ok<V4E_CFG_W64_1STG>(M, N, K)) return false;
            run_wmma_blocktile_gemm<V4E_CFG_W64_1STG>(d_A, d_B, d_C, M, N, K);
            return true;
    }
    return false;
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
    print_all_v4e_attrs();
    {
        int c = v4e_auto_choice(M, N, K);
        printf("  [auto] V4e-auto dispatches to: %s\n",
               c >= 0 ? V4E_NAMES[c] : "(none: shape not supported)");
    }

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
    float* h_C_v4e = (float*)malloc(size_C * sizeof(float));

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
    float* d_C_v4e;
    CUDA_CHECK(cudaMalloc(&d_C_v4e, size_C * sizeof(float)));

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

        for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
            // clear first so stale results from a previous config can't pass
            CUDA_CHECK(cudaMemset(d_C_v4e, 0, size_C * sizeof(float)));
            if (!run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K)) {
                printf("  [verify %s] skipped (shape not divisible)\n",
                       V4E_NAMES[cfg]);
                continue;
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_v4e, d_C_v4e, size_C * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            printf("  [verify %s]\n", V4E_NAMES[cfg]);
            verify_result(h_C_v4e, h_C_ref, (int)size_C, 0.5f, 0.05f);
        }

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

    // ---------- Benchmark: V4e tile sweep ----------
    double gflops_v4e[V4E_NUM_CFGS] = {};
    for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
        if (!run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K)) {
            printf("  [bench] %s: skipped (shape not divisible)\n",
                   V4E_NAMES[cfg]);
            continue;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = benchmark_kernel(
            [&]() { run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K); });
        gflops_v4e[cfg] = compute_gflops(M, N, K, ms);
        printf("  [bench] %s WMMA kernel: avg_time=%.3f ms, GFLOPS=%.2f\n",
               V4E_NAMES[cfg], ms, gflops_v4e[cfg]);
    }

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

    int best_cfg = -1;
    for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
        if (gflops_v4e[cfg] <= 0.0) continue;
        printf("  [compare] %s kernel achieving cuBLAS performance %.1f%% "
               "(%.2fx over tiled)\n",
               V4E_NAMES[cfg], 100.0 * gflops_v4e[cfg] / gflops_cublas,
               gflops_v4e[cfg] / gflops_tiled);
        if (best_cfg < 0 || gflops_v4e[cfg] > gflops_v4e[best_cfg]) best_cfg = cfg;
    }
    if (best_cfg >= 0)
        printf("  [sweep] best V4e config at N=%d: %s (%.1f%% of cuBLAS)\n", N,
               V4E_NAMES[best_cfg], 100.0 * gflops_v4e[best_cfg] / gflops_cublas);

    // ---------- Robust ratio: repeated rounds, cuBLAS measured around each ----------
    // Colab T4 clocks drift between and within runs (cuBLAS at N=2048 was
    // measured at both ~30 and ~40 TFLOPS). A single cuBLAS sample makes the
    // "% of cuBLAS" number unreliable, so each round measures cuBLAS right
    // before and right after the V4e configs and divides by their mean.
    // We report the median ratio over all rounds plus its min..max spread.
    const int kRounds = 7;
    std::vector<double> cub_gf;
    std::vector<std::vector<double>> ratio(V4E_NUM_CFGS);
    auto bench_cublas_gf = [&]() {
        float ms = benchmark_kernel([&]() {
            run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
        });
        return compute_gflops(M, N, K, ms);
    };
    for (int r = 0; r < kRounds; ++r) {
        double before = bench_cublas_gf();
        double gf[V4E_NUM_CFGS] = {};
        for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
            if (gflops_v4e[cfg] <= 0.0) continue;  // shape not supported
            float ms = benchmark_kernel(
                [&]() { run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K); });
            gf[cfg] = compute_gflops(M, N, K, ms);
        }
        double after = bench_cublas_gf();
        double cub = 0.5 * (before + after);
        cub_gf.push_back(before);
        cub_gf.push_back(after);
        for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg)
            if (gf[cfg] > 0.0) ratio[cfg].push_back(100.0 * gf[cfg] / cub);
    }
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };
    printf("  [robust] %d rounds; cuBLAS GFLOPS median=%.0f (min %.0f, max %.0f)\n",
           kRounds, median(cub_gf),
           *std::min_element(cub_gf.begin(), cub_gf.end()),
           *std::max_element(cub_gf.begin(), cub_gf.end()));
    for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
        if (ratio[cfg].empty()) continue;
        printf("  [robust] %-16s %% of cuBLAS: median=%.1f%% (min %.1f%%, max %.1f%%)\n",
               V4E_NAMES[cfg], median(ratio[cfg]),
               *std::min_element(ratio[cfg].begin(), ratio[cfg].end()),
               *std::max_element(ratio[cfg].begin(), ratio[cfg].end()));
    }
    printf("  (Acceptance Criteria: N>=1024 >= 70%% Deemed to meet the standard.)\n");
    printf("  (Judge acceptance on the [robust] median, not the single-shot [compare].)\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_dbuf));
    CUDA_CHECK(cudaFree(d_C_warptile));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaFree(d_C_v4e));
    free(h_C_v4e);
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
