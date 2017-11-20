//
// Created by yanhao on 17-11-20.
//

#ifndef PROJECT_MUL_BLAS_SSE_H
#define PROJECT_MUL_BLAS_SSE_H

float XR_gemm(const float *a, const float *b, int len);

int run_gemm(const float *A, const float *B, float *C, int M, int K, int N, int mode);

#endif //PROJECT_MUL_BLAS_SSE_H
