// register_blocking_gemm.cu
// 2D register-blocking SGEMM:  C[M,N] = A[M,K] * B[K,N]   (row-major, FP32)
//
// Build & self-test:  nvcc -O3 -arch=sm_75 register_blocking_gemm.cu -o gemm && ./gemm
// (sm_75 = T4; change -arch to match your GPU. Define NO_TEST_MAIN to drop main().)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err__), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Tile sizes are template parameters (not global macros), checked at compile time.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
register_blocking_gemm_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                                    float* __restrict__ C,
                              int M, int N, int K) {
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
    constexpr int strideA     = NUM_THREADS / BK;  // rows of A loaded per pass
    constexpr int strideB     = NUM_THREADS / BN;  // rows of B loaded per pass
    constexpr int A_LD        = BM + 4;            // padded leading dim of transposed As

    static_assert(BM % TM == 0 && BN % TN == 0, "BM/BN must be multiples of TM/TN");
    static_assert(NUM_THREADS % BK == 0, "NUM_THREADS must be a multiple of BK");
    static_assert(NUM_THREADS % BN == 0, "NUM_THREADS must be a multiple of BN (else strideB == 0)");
    static_assert(BM % strideA == 0, "BM must be a multiple of NUM_THREADS/BK");
    static_assert(BK % strideB == 0, "BK must be a multiple of NUM_THREADS/BN");
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BN % 4 == 0, "float4 shared-memory reads");

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // 1D thread index -> (BM/TM) x (BN/TN) thread grid
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // As is stored transposed (As[k][m]) so the inner loop reads TM contiguous floats.
    // +4 padding: conflict-free transposed stores, and rows stay 16-byte aligned.
    __shared__ __align__(16) float As[BK * A_LD];
    __shared__ __align__(16) float Bs[BK * BN];

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load BM x BK tile of A (transposed into As), zero-padding out-of-range ----
        #pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            const int r = rowBase + innerRowA + off;
            const int c = k0 + innerColA;
            As[innerColA * A_LD + innerRowA + off] =
                (r < M && c < K) ? A[static_cast<size_t>(r) * K + c] : 0.0f;
        }
        // ---- Load BK x BN tile of B, zero-padding out-of-range ----
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            const int r = k0 + innerRowB + off;
            const int c = colBase + innerColB;
            Bs[(innerRowB + off) * BN + innerColB] =
                (r < K && c < N) ? B[static_cast<size_t>(r) * N + c] : 0.0f;
        }
        __syncthreads();

        // ---- Outer-product accumulation in registers ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &As[k * A_LD + threadRow * TM + i]);
                regM[i] = v.x; regM[i + 1] = v.y; regM[i + 2] = v.z; regM[i + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TN; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &Bs[k * BN + threadCol * TN + i]);
                regN[i] = v.x; regN[i + 1] = v.y; regN[i + 2] = v.z; regN[i + 3] = v.w;
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m * TN + n] = fmaf(regM[m], regN[n], acc[m * TN + n]);
                }
            }
        }
        __syncthreads();
    }

    // ---- Write back with bounds check (grid uses ceil-div, so edge blocks exist) ----
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int row = rowBase + threadRow * TM + m;
        if (row >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int col = colBase + threadCol * TN + n;
            if (col < N) {
                C[static_cast<size_t>(row) * N + col] = acc[m * TN + n];
            }
        }
    }
}

void run_register_blocking_gemm(const float* d_A,
                                const float* d_B,
                                      float* d_C,
                                int M, int N, int K,
                                cudaStream_t stream = 0) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);  // 256

    if (M <= 0 || N <= 0 || K <= 0) return;

    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);    // ceil-div: covers the edges
    register_blocking_gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

#ifndef NO_TEST_MAIN
// Self-test against a CPU reference, using sizes that are NOT multiples of the tiles.
int main() {
    const int M = 1000, N = 777, K = 129;
    std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N),
                       hC(static_cast<size_t>(M) * N), ref(static_cast<size_t>(M) * N);
    srand(0);
    for (auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += double(hA[size_t(i) * K + k]) * hB[size_t(k) * N + j];
            ref[size_t(i) * N + j] = static_cast<float>(s);
        }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    run_register_blocking_gemm(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) maxErr = fmax(maxErr, fabs(hC[i] - ref[i]));
    std::printf("M=%d N=%d K=%d  max abs error = %.3e  -> %s\n",
                M, N, K, maxErr, maxErr < 1e-3 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return maxErr < 1e-3 ? 0 : 1;
}
#endif
