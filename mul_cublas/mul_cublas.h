//
// Created by yanhao on 17-11-21.
//

#ifndef PROJECT_MUL_CUBLAS_H
#define PROJECT_MUL_CUBLAS_H

#include <cublas_v2.h> //cuda自带库函数

extern "C" {
void run22(const cublasHandle_t &handle, const cudaStream_t &stream, float *a, float *b, float *c);
cublasStatus_t
addWithCuda(const cublasHandle_t &handle, float *c, const float *a, const float *b, unsigned int WA, unsigned int HA,
            unsigned int WB,
            unsigned int HB);
cublasStatus_t
addWithCuda5(const cublasHandle_t &handle, float *c,const float *a, const float *b, unsigned int WA, unsigned int HA,
             unsigned int WB,
             unsigned int HB);

cublasStatus_t
addWithCuda6(const cublasHandle_t &handle, const float *a, const float *b, const int WA, const int HA,
             const int WB,
             const int HB, float *c);

void im2col2(const float *data_im, const int channels, int height, int width, const int kszie,
             const int pad, const int stride,
             float *data_col, int *h_out, int *w_out);


}
#endif //PROJECT_MUL_CUBLAS_H
