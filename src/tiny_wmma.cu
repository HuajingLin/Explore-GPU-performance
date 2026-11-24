//tiny_wmma.cu

#include <mma.h>
#include <cstdio>
using namespace nvcuda;

__global__ void tiny_wmma_gemm(const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);   // all 32 lanes cooperate
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    // peek inside: which elements does each lane hold?
    for(int i = 0; i< c.num_elements; i++){
        printf("lane %2d: c.x[%d] = %6.1f\n", threadIdx.x, i, c.x[i]);
    }
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main(){
    half *A, *B;
    float *C;
    cudaMallocManaged(&A, 16*16*sizeof(half));
    cudaMallocManaged(&B, 16*16*sizeof(half));
    cudaMallocManaged(&C, 16*16*sizeof(float));
    for(int i=0; i<16*16; i++){
        A[i] = __float2half(1.0f);
        //B[i] = __float2half(1.0f);
    }
    for (int r=0; r<16; r++){
        for (int c=0; c<16; c++){
            B[r][c] = r*16 + c;
        }
    }
    tiny_wmma_gemm<<<1, 32>>>(A, B, C);
    cudaDeviceSynchronize();
    for(int i=0; i<16*16; i++){
        printf("C[%d] = %6.1f\n", i, C[i]);
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}