//
// Created by yanhao on 17-11-21.
//

#ifndef PROJECT_MUL_CUBLAS_H
#define PROJECT_MUL_CUBLAS_H
#include <cublas_v2.h> //cuda自带库函数

extern "C" {
void run(const cublasHandle_t &handle,const cudaStream_t&stream,float *a,float*b,float *c);
cublasStatus_t addWithCuda(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                        unsigned int HB);
cublasStatus_t addWithCuda5(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                            unsigned int HB);
}
#endif //PROJECT_MUL_CUBLAS_H
