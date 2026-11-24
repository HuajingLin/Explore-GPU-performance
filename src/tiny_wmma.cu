// Minimal WMMA experiment: one warp computes a single 16x16x16 tile.
//
// Build and run (T4 / sm_75):
//   nvcc -O2 -arch=sm_75 tiny_wmma.cu -o tiny_wmma && ./tiny_wmma
//
// A = identity, B[r][c] = r * 16 + c, so C = A * B = B. Every value a lane
// prints is r * 16 + c, which tells you exactly which element of C that lane
// holds: row = value / 16, col = value % 16.

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(err_));              \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int kN = 16;  // one WMMA tile: 16 x 16 x 16

__global__ void tiny_wmma(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;

    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, kN);  // all 32 lanes must call this together
    wmma::load_matrix_sync(b, B, kN);
    wmma::mma_sync(c, a, b, c);

    // Peek inside the accumulator: which elements of C does each lane hold?
    // The layout is not guaranteed by the WMMA API; this is only to learn.
    for (int i = 0; i < c.num_elements; ++i) {
        const int v = static_cast<int>(c.x[i]);
        printf("lane %2d  c.x[%d] = %3d  -> C[%2d][%2d]\n", threadIdx.x, i, v,
               v / kN, v % kN);
    }

    wmma::store_matrix_sync(C, c, kN, wmma::mem_row_major);
}

int main() {
    // ---- host inputs ----
    half  hA[kN * kN], hB[kN * kN];
    float hC[kN * kN];
    for (int r = 0; r < kN; ++r)
        for (int c = 0; c < kN; ++c) {
            hA[r * kN + c] = __float2half(r == c ? 1.0f : 0.0f);  // identity
            // 0..255 are integers, exactly representable in fp16 (up to 2048)
            hB[r * kN + c] = __float2half(static_cast<float>(r * kN + c));
        }

    // ---- device buffers (cudaMalloc gives 256-byte alignment, enough for
    //      WMMA's 32-byte pointer requirement) ----
    half  *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    // ---- one block of exactly one warp ----
    tiny_wmma<<<1, 32>>>(dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // also flushes the device printf

    // ---- check C == B ----
    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < kN * kN; ++i)
        if (hC[i] != __half2float(hB[i])) ++bad;
    std::printf("\nC == B: %s (%d mismatches)\n", bad ? "NO" : "YES", bad);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return bad ? EXIT_FAILURE : EXIT_SUCCESS;
}