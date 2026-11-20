// sgemm_vectorized_dbuf.cu
//
// Single-precision GEMM with shared-memory double buffering and float4
// vectorized global/shared memory access.
//
//   C = alpha * A * B + beta * C      (BLAS-style SGEMM semantics)
//
// Layout: all matrices are row-major and densely packed.
//   A: M x K (leading dim K), B: K x N (leading dim N), C: M x N (leading dim N)
//
// Requirements (checked on the host before launch):
//   M % BM == 0, N % BN == 0, K % BK == 0, and A/B/C are 16-byte aligned.
//
// Build & self-test (e.g. on a Colab T4):
//   nvcc -O3 -arch=sm_75 -DSGEMM_SELF_TEST sgemm_vectorized_dbuf.cu -o sgemm && ./sgemm

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vectorized_dbuf_kernel(int M, int N, int K, float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    // ------------------------------------------------------------------
    // Compile-time configuration and sanity checks
    // ------------------------------------------------------------------
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

    // As is stored transposed (K-major). Padding by 4 floats removes the
    // bank conflicts of the transposed store while keeping every row
    // 16-byte aligned for float4 reads.
    constexpr int AS_LD = BM + 4;

    // Global -> shared load mapping (one float4 per thread per pass)
    constexpr int A_THREADS_PER_ROW = BK / 4;                        // float4s per row of the A tile
    constexpr int A_ROW_STRIDE      = NUM_THREADS / A_THREADS_PER_ROW;
    constexpr int A_LOADS           = BM / A_ROW_STRIDE;             // passes needed for the A tile
    constexpr int B_THREADS_PER_ROW = BN / 4;                        // float4s per row of the B tile
    constexpr int B_ROW_STRIDE      = NUM_THREADS / B_THREADS_PER_ROW;
    constexpr int B_LOADS           = BK / B_ROW_STRIDE;             // passes needed for the B tile

    // Compute mapping: each thread owns a TM x TN micro-tile.
    // Along N the micro-tile is split into TN/4 float4 chunks spaced
    // N_CHUNK_STRIDE apart, so that consecutive threads read consecutive
    // float4s from Bs (conflict-free LDS.128) and write consecutive
    // float4s to C (fully coalesced stores).
    constexpr int THREADS_PER_ROW_C = BN / TN;
    constexpr int TN_VEC            = TN / 4;
    constexpr int N_CHUNK_STRIDE    = THREADS_PER_ROW_C * 4;

    static_assert(NUM_THREADS > 0 && NUM_THREADS <= 1024, "Invalid thread count");
    static_assert(BM % TM == 0 && BN % TN == 0, "Block tile must be divisible by thread tile");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "TM/TN must be multiples of 4 for float4 access");
    static_assert(BK % 4 == 0 && BN % 4 == 0, "BK/BN must be multiples of 4 for float4 loads");
    static_assert(NUM_THREADS % A_THREADS_PER_ROW == 0 && BM % A_ROW_STRIDE == 0,
                  "Thread count does not evenly cover the A tile");
    static_assert(NUM_THREADS % B_THREADS_PER_ROW == 0 && BK % B_ROW_STRIDE == 0,
                  "Thread count does not evenly cover the B tile");
    static_assert(AS_LD % 4 == 0, "Padded As row must stay 16-byte aligned");
    static_assert(2 * BK * (AS_LD + BN) * sizeof(float) <= 48 * 1024,
                  "Static shared memory exceeds 48 KB");

    // ------------------------------------------------------------------
    // Shared memory (double buffered)
    // ------------------------------------------------------------------
    __shared__ __align__(16) float As[2][BK][AS_LD];  // transposed: As[k][m]
    __shared__ __align__(16) float Bs[2][BK][BN];     // natural:    Bs[k][n]

    const int tid       = threadIdx.x;
    const int threadRow = tid / THREADS_PER_ROW_C;
    const int threadCol = tid % THREADS_PER_ROW_C;

    const int aRow = tid / A_THREADS_PER_ROW;
    const int aCol = (tid % A_THREADS_PER_ROW) * 4;
    const int bRow = tid / B_THREADS_PER_ROW;
    const int bCol = (tid % B_THREADS_PER_ROW) * 4;

    // Move base pointers to this block's tile (size_t avoids int overflow)
    A += static_cast<size_t>(blockIdx.y) * BM * K;
    B += static_cast<size_t>(blockIdx.x) * BN;
    C += static_cast<size_t>(blockIdx.y) * BM * N + static_cast<size_t>(blockIdx.x) * BN;

    // Register staging buffers for the next tile (global -> reg -> smem)
    float4 ldgA[A_LOADS];
    float4 ldgB[B_LOADS];

    // Issue global loads of one K-tile into registers
    auto loadTileToRegs = [&](int kOffset) {
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int r = aRow + i * A_ROW_STRIDE;
            ldgA[i] = *reinterpret_cast<const float4*>(
                A + static_cast<size_t>(r) * K + kOffset + aCol);
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int r = bRow + i * B_ROW_STRIDE;
            ldgB[i] = *reinterpret_cast<const float4*>(
                B + static_cast<size_t>(kOffset + r) * N + bCol);
        }
    };

    // Write staged registers into the given shared-memory buffer
    auto storeRegsToSmem = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            const int r = aRow + i * A_ROW_STRIDE;
            As[buf][aCol + 0][r] = ldgA[i].x;   // transpose on store
            As[buf][aCol + 1][r] = ldgA[i].y;
            As[buf][aCol + 2][r] = ldgA[i].z;
            As[buf][aCol + 3][r] = ldgA[i].w;
        }
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            const int r = bRow + i * B_ROW_STRIDE;
            *reinterpret_cast<float4*>(&Bs[buf][r][bCol]) = ldgB[i];
        }
    };

    float acc[TM][TN] = {};
    float regM[TM];
    float regN[TN];

    // ------------------------------------------------------------------
    // Prologue: fill buffer 0
    // ------------------------------------------------------------------
    loadTileToRegs(0);
    storeRegsToSmem(0);
    __syncthreads();

    // ------------------------------------------------------------------
    // Main loop
    // One barrier per iteration is sufficient:
    //  - RAW: the barrier after writing buf^1 makes it visible before it is read.
    //  - WAR: buf is overwritten only in the next iteration, i.e. after that
    //         same barrier, by which time every thread has finished reading it.
    // ------------------------------------------------------------------
    const int numTiles = K / BK;
    for (int t = 0; t < numTiles; ++t) {
        const int  buf     = t & 1;
        const bool hasNext = (t + 1) < numTiles;

        // Start the next tile's global loads early; their latency is hidden
        // behind the FMAs below (the registers are consumed only afterwards).
        if (hasNext) {
            loadTileToRegs((t + 1) * BK);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // A fragment: TM contiguous values (same address within a half-warp -> broadcast)
            #pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&As[buf][k][threadRow * TM + i]);
                regM[i + 0] = v.x;
                regM[i + 1] = v.y;
                regM[i + 2] = v.z;
                regM[i + 3] = v.w;
            }
            // B fragment: TN/4 float4 chunks, interleaved across threads
            #pragma unroll
            for (int c = 0; c < TN_VEC; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &Bs[buf][k][c * N_CHUNK_STRIDE + threadCol * 4]);
                regN[c * 4 + 0] = v.x;
                regN[c * 4 + 1] = v.y;
                regN[c * 4 + 2] = v.z;
                regN[c * 4 + 3] = v.w;
            }
            // Outer product
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] = fmaf(regM[m], regN[n], acc[m][n]);
                }
            }
        }

        if (hasNext) {
            storeRegsToSmem(buf ^ 1);
            __syncthreads();
        }
    }

    // ------------------------------------------------------------------
    // Epilogue: C = alpha * acc + beta * C (C is not read when beta == 0,
    // so uninitialized/NaN contents of C do not propagate)
    // ------------------------------------------------------------------
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = threadRow * TM + m;
        #pragma unroll
        for (int c = 0; c < TN_VEC; ++c) {
            const int col = c * N_CHUNK_STRIDE + threadCol * 4;
            float4* dst = reinterpret_cast<float4*>(C + static_cast<size_t>(row) * N + col);
            const float* r = &acc[m][c * 4];

            float4 out;
            if (beta == 0.0f) {
                out = make_float4(alpha * r[0], alpha * r[1], alpha * r[2], alpha * r[3]);
            } else {
                const float4 old = *dst;
                out.x = fmaf(alpha, r[0], beta * old.x);
                out.y = fmaf(alpha, r[1], beta * old.y);
                out.z = fmaf(alpha, r[2], beta * old.z);
                out.w = fmaf(alpha, r[3], beta * old.w);
            }
            *dst = out;
        }
    }
}

// ----------------------------------------------------------------------
// Host launcher: validates arguments, launches, and reports launch errors.
// ----------------------------------------------------------------------
static inline bool is_aligned_16(const void* p)
{
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

cudaError_t run_vectorized_dbuf_gemm(int M, int N, int K, float alpha,
                                     const float* d_A, const float* d_B,
                                     float beta, float* d_C,
                                     cudaStream_t stream = 0)
{
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

    if (M < 0 || N < 0 || K < 0) return cudaErrorInvalidValue;
    if (M == 0 || N == 0)        return cudaSuccess;          // nothing to do
    if (M % BM || N % BN || K % BK || K == 0) {
        fprintf(stderr, "sgemm: M,N must be multiples of %d/%d and K a positive multiple of %d\n",
                BM, BN, BK);
        return cudaErrorInvalidValue;
    }
    if (!is_aligned_16(d_A) || !is_aligned_16(d_B) || !is_aligned_16(d_C)) {
        fprintf(stderr, "sgemm: A, B and C must be 16-byte aligned\n");
        return cudaErrorMisalignedAddress;
    }

    const dim3 block(NUM_THREADS);
    const dim3 grid(N / BN, M / BM);
    sgemm_vectorized_dbuf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    return cudaGetLastError();
}

// ----------------------------------------------------------------------
// Optional self-test against a CPU reference (compile with -DSGEMM_SELF_TEST)
// ----------------------------------------------------------------------
#ifdef SGEMM_SELF_TEST
#include <cmath>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

int main()
{
    const int M = 256, N = 384, K = 264;
    const float alpha = 1.5f, beta = 0.5f;

    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N);
    srand(42);
    for (auto& x : hA) x = rand() / float(RAND_MAX) - 0.5f;
    for (auto& x : hB) x = rand() / float(RAND_MAX) - 0.5f;
    for (auto& x : hC) x = rand() / float(RAND_MAX) - 0.5f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(run_vectorized_dbuf_gemm(M, N, K, alpha, dA, dB, beta, dC));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(hC.size());
    CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) ref += double(hA[size_t(i) * K + k]) * hB[size_t(k) * N + j];
            ref = alpha * ref + beta * hC[size_t(i) * N + j];
            maxErr = fmax(maxErr, fabs(ref - out[size_t(i) * N + j]));
        }
    printf("max abs error = %.3e -> %s\n", maxErr, maxErr < 1e-3 ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return maxErr < 1e-3 ? 0 : 1;
}
#endif
