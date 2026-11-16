// v4_wmma_tensorcore.cu
//
// Build (Colab T4, sm_75):
//   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
// Run:
//   ./build/v4_wmma 512     # verifies every kernel vs the CPU reference AND cuBLAS
//   ./build/v4_wmma 1024    # N > 512: every kernel is verified against cuBLAS only
//   ./build/v4_wmma 2048
//   ./build/v4_wmma 4096
// Soak (one kernel only, sustained load, GFLOPS printed every ~0.5 s; use it
// together with an nvidia-smi log to see steady-state clock / power):
//   ./build/v4_wmma 2048 soak v4e 6      # v4e | 128x64 | w64 | w64s | cublas, seconds
// Two ratios are reported; they answer different questions:
//   [robust]    equal-CLOCK ratio. Short interleaved windows, config order
//               rotated, so every kernel runs at the same GPU clock. This is
//               kernel efficiency, and what the >= 70% criterion is judged on.
//   [sustained] equal-POWER ratio. Each kernel alone under sustained load;
//               on the 70 W-capped T4 the driver settles on a different clock
//               for each kernel. This is the throughput you actually get.
// The single-shot [bench]/[compare] lines are only a first look: clocks are
// still moving while they are measured.
// The [attr] lines report regs / spills / smem / blocks-per-SM per config.

#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
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

// ---------- verification tolerances ----------
// Both references (CPU GEMM and cuBLAS) see exactly the same FP16-rounded
// inputs as the kernels, so the only remaining difference is FP32 summation
// order. That lets the tolerance be ~50x tighter than the old 0.5 / 5%,
// tight enough that a single dropped 16-wide K step fails.
// If verify_result reports failures on kernels you trust (it lives in
// common.cuh), loosen these a little rather than going back to 0.5 / 5%.
static const float kVerifyAtol = 1e-2f;
static const float kVerifyRtol = 1e-3f;

// Abort unless (M, N, K) is an exact multiple of the kernel's block shape.
// None of the kernels in this file handle partial tiles: a ragged edge
// would make load_matrix_sync / store_matrix_sync touch out-of-bounds memory.
static void require_shape(const char* name, int M, int N, int K, int bm,
                          int bn, int bk) {
    if (M <= 0 || N <= 0 || K <= 0 || M % bm != 0 || N % bn != 0 ||
        K % bk != 0) {
        fprintf(stderr,
                "ERROR [%s]: M=%d N=%d K=%d must be positive multiples of "
                "%d / %d / %d\n",
                name, M, N, K, bm, bn, bk);
        exit(EXIT_FAILURE);
    }
}

// Runs `launch` continuously: `warm_s` seconds unmeasured (lets a
// power-capped GPU settle on its clock for this kernel), then `meas_s`
// seconds measured. Returns the average time per launch in ms.
template <typename Launch>
double sustained_ms_per_launch(Launch launch, double warm_s, double meas_s) {
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const int kBatch = 20;
    auto run_for = [&](double s) {
        const auto t0 = clk::now();
        int launches = 0;
        do {
            for (int i = 0; i < kBatch; ++i) launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            launches += kBatch;
        } while (secs(t0, clk::now()) < s);
        return secs(t0, clk::now()) * 1000.0 / launches;
    };
    run_for(warm_s);
    return run_for(meas_s);
}

// Number of SMs on the current device (40 on a Tesla T4), queried once.
static int device_num_sms() {
    static int cached = 0;
    if (cached == 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &cached, cudaDevAttrMultiProcessorCount, dev));
    }
    return cached;
}

// ---------- FP16 back to FP32 (host), for the CPU reference ----------
void convert_to_float(const half* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = __half2float(src[i]);
    }
}

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
    const int blockM = WMMA_M * (blockDim.x / 32);  // 4 warps along M -> 64 rows
    const int blockN = WMMA_N * blockDim.y;         // 4 warps along N -> 64 cols
    require_shape("naive WMMA", M, N, K, blockM, blockN, WMMA_K);
    dim3 gridDim(M / blockM, N / blockN);
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
    require_shape("tiled WMMA", M, N, K, TILE_M, TILE_N, TILE_K);
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
    require_shape("tiled+dbuf WMMA", M, N, K, TILE_M, TILE_N, TILE_K);
    dim3 gridDim(M / TILE_M, N / TILE_N);
    wmma_tiled_dbuf_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

// ---------- V4c: Shared Memory Tiling + warp-level multi-fragment tiling ----------
//
// Each warp owns a (WARP_TILE_M*16) x (WARP_TILE_N*16) = 64x32 output region,
// i.e. 4x2 = 8 accumulator fragments. With a 64x64 block tile that means
// 1 warp along M and 2 warps along N: 2 warps = 64 threads per block.
#define WARP_TILE_M 4  // number of WMMA_M tiles each warp computes, M direction
#define WARP_TILE_N 2  // number of WMMA_N tiles each warp computes, N direction
#define WARPS_M (TILE_M / (WARP_TILE_M * WMMA_M))  // = 64/64 = 1 warp along M
#define WARPS_N (TILE_N / (WARP_TILE_N * WMMA_N))  // = 64/32 = 2 warps along N
static_assert(WARPS_M >= 1 && WARPS_M * WARP_TILE_M * WMMA_M == TILE_M,
              "warp tiles must exactly cover TILE_M");
static_assert(WARPS_N >= 1 && WARPS_N * WARP_TILE_N * WMMA_N == TILE_N,
              "warp tiles must exactly cover TILE_N");

__global__ void wmma_warptile_gemm_kernel(const half* __restrict__ A,
                                           const half* __restrict__ B,
                                           float* __restrict__ C, int M,
                                           int N, int K) {
    int cRow = blockIdx.x;  // block's M-direction tile index
    int cCol = blockIdx.y;  // block's N-direction tile index

    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // flattened thread id within block
    int totalThreads = blockDim.x * blockDim.y;         // WARPS_M*WARPS_N*32 = 64

    int warpM_local = threadIdx.x / warpSize;  // 0..WARPS_M-1 (always 0)
    int warpN_local = threadIdx.y;             // 0..WARPS_N-1 (0..1)

    // This warp's output region, in local (block-relative) coordinates
    int warpRowBase = warpM_local * WARP_TILE_M * WMMA_M;  // always 0
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
        // WARP_TILE_N cross-product of mma_sync calls: 4 + 2 = 6 fragment
        // loads feed 8 MMAs, versus 2 loads per MMA in V4b ----
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
    // blockDim: WARPS_M*32 threads along x (1 warp x 32 = 32), WARPS_N
    // along y (2) -> 64 threads = 2 warps per block, covering the same
    // TILE_M x TILE_N = 64x64 output region as V4b/V4d, just with 2 "fat"
    // warps (64x32 each) instead of 16 "thin" ones (16x16 each).
    dim3 blockDim(WARPS_M * 32, WARPS_N);
    require_shape("warptile WMMA", M, N, K, TILE_M, TILE_N, TILE_K);
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
// kSpread = 0: all global loads of the next tile are issued back to back at
//              the top of the K step.
// kSpread = 1: the same loads are issued in small groups, one group before
//              each row of MMAs, to relieve the global-memory pipe
//              (ncu: lg_throttle was 0.53 here vs 0.12 in cuBLAS).
template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2, int kSpread = 0>
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

    // kSpread: the K step has kLoadSlots = (kBK/16) * kFragM rows of MMAs.
    // Global load number n (A vectors first, then B vectors) goes to slot
    // n * kLoadSlots / kLoadsPerT. `slot` is a constant after unrolling, so
    // the tests below fold away and ra[]/rb[] stay in registers.
    constexpr int kLoadsPerT = kAPerT + kBPerT;
    constexpr int kLoadSlots = (kBK / WMMA_K) * kFragM;
    auto load_global_slot = [&](int k0, int slot) {
#pragma unroll
        for (int i = 0; i < kAPerT; ++i) {
            if ((i * kLoadSlots) / kLoadsPerT != slot) continue;
            int idx = tid + i * kThreads;
            int r = idx / kAVecsPerRow;
            int c = (idx % kAVecsPerRow) * 8;
            ra[i] = *reinterpret_cast<const int4*>(Ag + (size_t)r * K + k0 + c);
        }
#pragma unroll
        for (int i = 0; i < kBPerT; ++i) {
            if (((kAPerT + i) * kLoadSlots) / kLoadsPerT != slot) continue;
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

        // kSpread == 0: issue the next tile's global loads first; latency
        // overlaps the MMAs.
        if (kSpread == 0 && hasNext) load_global(k0 + kBK);
        // kSpread == 1: the loads are issued between the MMA rows below.
        // Kept branch-free: on the last K step they re-read the current
        // tile, and that result is never stored.
        const int kNext = hasNext ? k0 + kBK : k0;

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
            for (int i = 0; i < kFragM; ++i) {
                if (kSpread != 0)
                    load_global_slot(kNext, (kk / WMMA_K) * kFragM + i);
#pragma unroll
                for (int j = 0; j < kFragN; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
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
          int kStages = 2, int kSpread = 0>
bool v4e_shape_ok(int M, int N, int K) {
    return (M % kBM == 0) && (N % kBN == 0) && (K % kBK == 0);
}

template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2, int kSpread = 0>
void run_wmma_blocktile_gemm(const half* d_A, const half* d_B, float* d_C,
                             int M, int N, int K) {
    constexpr int kThreads = (kBM / kWM) * (kBN / kWN) * 32;
    require_shape("V4e", M, N, K, kBM, kBN, kBK);
    dim3 block(kThreads);
    dim3 grid(N / kBN, M / kBM);
    wmma_blocktile_gemm_kernel<kBM, kBN, kBK, kWM, kWN, kMinBlocks, kStages, kSpread>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

// Prints registers, local memory (spills), static smem and how many blocks
// fit per SM, as reported by the runtime for the compiled kernel.
template <int kBM, int kBN, int kBK, int kWM, int kWN, int kMinBlocks = 1,
          int kStages = 2, int kSpread = 0>
void print_v4e_attrs(const char* name) {
    auto kern = wmma_blocktile_gemm_kernel<kBM, kBN, kBK, kWM, kWN, kMinBlocks, kStages, kSpread>;
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
//   V4e-w64-sprd : w64-1stg with spread global loads (244 regs). What the
//                  dispatcher uses for N >= 1024.
// Tried and removed (2026-11-15): shell-ordered fragment loads. Same
// instruction count and cycles as w64-sprd: the compiler decides where the
// LDSM instructions go, source order inside the WMMA API does not.
#define V4E_CFG_128x64 128, 64, 32, 64, 32
#define V4E_CFG_W64_1STG 128, 128, 32, 64, 64, 1, 1
// w64-1stg with the global loads spread between the MMA rows (kSpread = 1).
// ncu at N=2048 (2026-11-15): 745,016 cycles vs 756,547 for w64-1stg (-1.5%),
// 244 registers instead of 254, bit-identical output. The dispatcher uses it
// for the 128x128 case. V4e-w64-1stg is kept as the comparison baseline.
#define V4E_CFG_W64_SPRD 128, 128, 32, 64, 64, 1, 1, 1
#define V4E_NUM_CFGS 4
#define V4E_AUTO 3  // index of the size-based dispatcher
static const char* V4E_NAMES[V4E_NUM_CFGS] = {"V4e-128x64", "V4e-w64-1stg",
                                              "V4e-w64-sprd", "V4e-auto"};

void print_all_v4e_attrs() {
    print_v4e_attrs<V4E_CFG_128x64>(V4E_NAMES[0]);
    print_v4e_attrs<V4E_CFG_W64_1STG>(V4E_NAMES[1]);
    print_v4e_attrs<V4E_CFG_W64_SPRD>(V4E_NAMES[2]);
}

// Size-based dispatcher. If a 128x128 grid would give fewer blocks than the
// device has SMs (40 on a T4; queried at runtime), use the smaller 128x64
// tile so every SM gets work.
// Returns the config index (0 = 128x64, 2 = w64-sprd), or -1 if neither
// shape fits.
int v4e_auto_choice(int M, int N, int K) {
    const int kNumSMs = device_num_sms();
    bool w64_ok = v4e_shape_ok<V4E_CFG_W64_SPRD>(M, N, K);
    bool c64_ok = v4e_shape_ok<V4E_CFG_128x64>(M, N, K);
    if (w64_ok && (M / 128) * (N / 128) >= kNumSMs) return 2;
    if (c64_ok) return 0;
    if (w64_ok) return 2;
    return -1;
}

// Launch config `cfg` (0..V4E_NUM_CFGS-1; V4E_AUTO = size-based dispatcher). Returns false if the shape doesn't fit.
bool run_v4e_cfg(int cfg, const half* d_A, const half* d_B, float* d_C, int M,
                 int N, int K) {
    if (cfg == V4E_AUTO) cfg = v4e_auto_choice(M, N, K);
    switch (cfg) {
        case 0:
            if (!v4e_shape_ok<V4E_CFG_128x64>(M, N, K)) return false;
            run_wmma_blocktile_gemm<V4E_CFG_128x64>(d_A, d_B, d_C, M, N, K);
            return true;
        case 1:
            if (!v4e_shape_ok<V4E_CFG_W64_1STG>(M, N, K)) return false;
            run_wmma_blocktile_gemm<V4E_CFG_W64_1STG>(d_A, d_B, d_C, M, N, K);
            return true;
        case 2:
            if (!v4e_shape_ok<V4E_CFG_W64_SPRD>(M, N, K)) return false;
            run_wmma_blocktile_gemm<V4E_CFG_W64_SPRD>(d_A, d_B, d_C, M, N, K);
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

    // ---------- soak mode: one kernel, sustained load ----------
    // ./v4_wmma N soak <v4e|128x64|w64|w64s|cublas> [seconds]
    // The T4 is power-capped at 70 W; under sustained load the driver lowers
    // the clock. This mode shows each kernel's steady-state throughput on
    // its own, without other kernels heating/cooling the GPU in between.
    if (argc > 2 && strcmp(argv[2], "soak") == 0) {
        const char* target = (argc > 3) ? argv[3] : "v4e";
        const double seconds = (argc > 4) ? atof(argv[4]) : 6.0;
        cublasHandle_t soak_handle = nullptr;
        int cfg = -1;
        if (strcmp(target, "cublas") == 0) {
            CUBLAS_CHECK(cublasCreate(&soak_handle));
        } else if (strcmp(target, "128x64") == 0) {
            cfg = 0;
        } else if (strcmp(target, "w64") == 0) {
            cfg = 1;
        } else if (strcmp(target, "w64s") == 0) {
            cfg = 2;
        } else if (strcmp(target, "v4e") == 0) {
            cfg = V4E_AUTO;
        } else {
            fprintf(stderr, "usage: %s N soak <v4e|128x64|w64|w64s|cublas> [seconds]\n",
                    argv[0]);
            return EXIT_FAILURE;
        }
        auto launch = [&]() {
            if (soak_handle) {
                run_cublas_gemm(soak_handle, d_A, d_B, d_C_cublas, M, N, K);
            } else if (!run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K)) {
                fprintf(stderr, "ERROR: shape not supported by %s\n", target);
                exit(EXIT_FAILURE);
            }
        };
        using clk = std::chrono::steady_clock;
        auto secs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        printf("  [soak] target=%s, %.1f s\n", target, seconds);
        const int kBatch = 20;
        const auto t0 = clk::now();
        while (secs(t0, clk::now()) < seconds) {
            const auto w0 = clk::now();
            int launches = 0;
            do {
                for (int i = 0; i < kBatch; ++i) launch();
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                launches += kBatch;
            } while (secs(w0, clk::now()) < 0.5);
            const auto w1 = clk::now();
            const float ms = (float)(secs(w0, w1) * 1000.0 / launches);
            printf("  [soak] t=%5.1f s  %7.3f ms/GEMM  %8.0f GFLOPS\n",
                   secs(t0, w1), ms, compute_gflops(M, N, K, ms));
            fflush(stdout);
        }
        if (soak_handle) CUBLAS_CHECK(cublasDestroy(soak_handle));
        return 0;  // process exit releases host/device buffers
    }

    // ---------- verify ----------
    // Two references, both built from the SAME FP16-rounded inputs the
    // kernels see, so only FP32 summation order differs:
    //   (a) cuBLAS, for every N (cheap: one GEMM on the GPU);
    //   (b) CPU GEMM, for N <= 512 only (O(N^3) on the host). It also
    //       cross-checks cuBLAS itself, so (a) is not trusted blindly.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, size_C * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("  [verify] tolerances: atol=%g rtol=%g\n", kVerifyAtol,
           kVerifyRtol);
    float* h_C_ref = nullptr;
    int verify_failures = 0;  // any FAIL makes the program exit non-zero
    if (N <= 512) {
        float* h_A_q = (float*)malloc(size_A * sizeof(float));
        float* h_B_q = (float*)malloc(size_B * sizeof(float));
        h_C_ref = (float*)malloc(size_C * sizeof(float));
        convert_to_float(h_A_f16, h_A_q, size_A);
        convert_to_float(h_B_f16, h_B_q, size_B);
        cpu_gemm_ref(h_A_q, h_B_q, h_C_ref, M, N, K);
        free(h_A_q);
        free(h_B_q);
        printf("  [verify cuBLAS vs CPU ref]\n");
        if (!verify_result(h_C_cublas, h_C_ref, (int)size_C, kVerifyAtol,
                           kVerifyRtol))
            ++verify_failures;
    } else {
        printf("  [verify] N>512: CPU reference skipped, checking against "
               "cuBLAS only\n");
    }

    // Clears the output first (stale results can't pass), launches, checks
    // launch + execution errors, then compares against both references.
    // `launch` returns false if the kernel doesn't support this shape.
    auto verify_kernel = [&](const char* name, float* d_C, float* h_C,
                             auto&& launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, size_C * sizeof(float)));
        if (!launch()) {
            printf("  [verify %s] skipped (shape not divisible)\n", name);
            return;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(float),
                              cudaMemcpyDeviceToHost));
        if (h_C_ref) {
            printf("  [verify %s vs CPU ref]\n", name);
            if (!verify_result(h_C, h_C_ref, (int)size_C, kVerifyAtol,
                               kVerifyRtol))
                ++verify_failures;
        }
        printf("  [verify %s vs cuBLAS]\n", name);
        if (!verify_result(h_C, h_C_cublas, (int)size_C, kVerifyAtol,
                           kVerifyRtol))
            ++verify_failures;
    };

    verify_kernel("naive wmma kernel", d_C_wmma, h_C_wmma, [&]() {
        run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K);
        return true;
    });
    verify_kernel("tiled wmma kernel", d_C_tiled, h_C_tiled, [&]() {
        run_wmma_tiled_gemm(d_A, d_B, d_C_tiled, M, N, K);
        return true;
    });
    verify_kernel("tiled+dbuf wmma kernel", d_C_dbuf, h_C_dbuf, [&]() {
        run_wmma_tiled_dbuf_gemm(d_A, d_B, d_C_dbuf, M, N, K);
        return true;
    });
    verify_kernel("warptile wmma kernel", d_C_warptile, h_C_warptile, [&]() {
        run_wmma_warptile_gemm(d_A, d_B, d_C_warptile, M, N, K);
        return true;
    });
    for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg) {
        verify_kernel(V4E_NAMES[cfg], d_C_v4e, h_C_v4e, [&]() {
            return run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K);
        });
    }
    free(h_C_ref);
    if (verify_failures > 0)
        printf("  [verify] *** %d check(s) FAILED: the benchmark numbers below "
               "are for WRONG results ***\n",
               verify_failures);
    else
        printf("  [verify] all checks passed\n");

    // Single-shot [bench] lines: the same short window repeated 5 times;
    // median with min..max. `bst` holds the stats of the latest measurement.
    BenchStats bst = {0.0f, 0.0f, 0.0f};
    auto bench_median = [&](auto&& fn) {
        bst = benchmark_kernel_stats(fn);
        return bst.median_ms;
    };

    // ---------- Benchmark: naive wmma kernel ----------
    float avg_ms_wmma =
        bench_median([&]() { run_wmma_gemm(d_A, d_B, d_C_wmma, M, N, K); });
    CUDA_CHECK(cudaGetLastError());
    double gflops_wmma = compute_gflops(M, N, K, avg_ms_wmma);
    printf("  [bench] naive WMMA kernel: median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
           avg_ms_wmma, bst.min_ms, bst.max_ms,
           gflops_wmma);

    // ---------- Benchmark: shared memory tiled wmma kernel ----------
    float avg_ms_tiled = bench_median(
        [&]() { run_wmma_tiled_gemm(d_A, d_B, d_C_tiled, M, N, K); });
    CUDA_CHECK(cudaGetLastError());
    double gflops_tiled = compute_gflops(M, N, K, avg_ms_tiled);
    printf("  [bench] tiled WMMA kernel: median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
           avg_ms_tiled, bst.min_ms, bst.max_ms,
           gflops_tiled);
    printf("  [compare] tiled over naive promote: %.2fx\n",
           gflops_tiled / gflops_wmma);

    // ---------- Benchmark: tiled + double buffering wmma kernel ----------
    float avg_ms_dbuf = bench_median(
        [&]() { run_wmma_tiled_dbuf_gemm(d_A, d_B, d_C_dbuf, M, N, K); });
    CUDA_CHECK(cudaGetLastError());
    double gflops_dbuf = compute_gflops(M, N, K, avg_ms_dbuf);
    printf("  [bench] tiled+dbuf WMMA kernel: median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
           avg_ms_dbuf, bst.min_ms, bst.max_ms,
           gflops_dbuf);
    printf("  [compare] tiled+dbuf over tiled promote: %.2fx\n",
           gflops_dbuf / gflops_tiled);

    // ---------- Benchmark: warp-level multi-fragment tiling wmma kernel ----------
    float avg_ms_warptile = bench_median(
        [&]() { run_wmma_warptile_gemm(d_A, d_B, d_C_warptile, M, N, K); });
    CUDA_CHECK(cudaGetLastError());
    double gflops_warptile = compute_gflops(M, N, K, avg_ms_warptile);
    printf("  [bench] warptile WMMA kernel: median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
           avg_ms_warptile, bst.min_ms, bst.max_ms,
           gflops_warptile);
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
        float ms = bench_median(
            [&]() { run_v4e_cfg(cfg, d_A, d_B, d_C_v4e, M, N, K); });
        gflops_v4e[cfg] = compute_gflops(M, N, K, ms);
        printf("  [bench] %s WMMA kernel: median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
               V4E_NAMES[cfg], ms, bst.min_ms, bst.max_ms,
               gflops_v4e[cfg]);
    }

    // ---------- Benchmark: cuBLAS run Tensor Core----------
    // (handle was created in the verify section above)

    float avg_ms_cublas = bench_median([&]() {
        run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K);
    });
    double gflops_cublas = compute_gflops(M, N, K, avg_ms_cublas);
    printf("  [bench] cuBLAS(TensorCore): median_time=%.3f ms (min %.3f, max %.3f), GFLOPS=%.2f\n",
           avg_ms_cublas, bst.min_ms, bst.max_ms,
           gflops_cublas);

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
    //
    // The order of the V4e configs is rotated every round. With a fixed
    // order the same kernel scored 69% in 2nd place and 75% in 5th place
    // (N=2048, 2026-11-15): the power-capped T4 changes clock within a
    // round, so position mattered more than the kernel. With rotation and
    // kRounds = 2 x (number of configs), every config sits in every position
    // exactly twice. Expect a wider min..max than before; compare medians.
    std::vector<int> valid_cfgs;
    for (int cfg = 0; cfg < V4E_NUM_CFGS; ++cfg)
        if (gflops_v4e[cfg] > 0.0) valid_cfgs.push_back(cfg);
    const int kNumValid = (int)valid_cfgs.size();
    const int kRounds = kNumValid > 0 ? 2 * kNumValid : 1;
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
        for (int i = 0; i < kNumValid; ++i) {
            const int cfg = valid_cfgs[(i + r) % kNumValid];  // rotated order
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
    printf("  [robust] equal-CLOCK ratio (kernel efficiency): short "
           "interleaved windows, same GPU clock for all\n");
    printf("  [robust] %d rounds, config order rotated; cuBLAS GFLOPS median=%.0f (min %.0f, max %.0f)\n",
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

    // ---------- Sustained ratio: each kernel alone, at its own steady clock ----------
    // On a power-capped GPU (T4: 70 W) the driver lowers the clock until the
    // power fits, and a kernel that draws more power per cycle is clocked
    // lower. Measured 2026-11-15 at N=2048: V4e ~850 MHz, cuBLAS ~655 MHz,
    // both ~69 W, throughput within 4%. So this ratio is usually much higher
    // than [robust]; both are correct, they answer different questions.
    {
        const int c = v4e_auto_choice(M, N, K);
        if (c >= 0) {
            const double kWarmS = 1.0, kMeasS = 2.0;
            const double ms_v4e = sustained_ms_per_launch(
                [&]() { run_v4e_cfg(V4E_AUTO, d_A, d_B, d_C_v4e, M, N, K); },
                kWarmS, kMeasS);
            // idle gap so the second measurement starts from a rested GPU
            std::this_thread::sleep_for(std::chrono::seconds(2));
            const double ms_cub = sustained_ms_per_launch(
                [&]() { run_cublas_gemm(handle, d_A, d_B, d_C_cublas, M, N, K); },
                kWarmS, kMeasS);
            const double gf_v4e = compute_gflops(M, N, K, (float)ms_v4e);
            const double gf_cub = compute_gflops(M, N, K, (float)ms_cub);
            printf("  [sustained] equal-POWER ratio (throughput on this GPU): "
                   "each kernel alone, %.0f s warm-up + %.0f s measured\n",
                   kWarmS, kMeasS);
            printf("  [sustained] V4e-auto (%s): %.0f GFLOPS   cuBLAS: %.0f "
                   "GFLOPS   -> %.1f%% of cuBLAS\n",
                   V4E_NAMES[c], gf_v4e, gf_cub, 100.0 * gf_v4e / gf_cub);
            printf("  (Not the acceptance number. It depends on this GPU's "
                   "power cap; [robust] does not.)\n");
        }
    }

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

    if (verify_failures > 0) {
        fprintf(stderr, "ERROR: %d verification check(s) failed\n",
                verify_failures);
        return EXIT_FAILURE;
    }
    return 0;
}
