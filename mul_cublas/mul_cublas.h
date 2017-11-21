//
// Created by yanhao on 17-11-21.
//

#ifndef PROJECT_MUL_CUBLAS_H
#define PROJECT_MUL_CUBLAS_H
#include <cublas_v2.h> //cuda自带库函数

extern "C" {
cublasStatus_t addWithCuda(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                        unsigned int HB);
cublasStatus_t addWithCuda2(const cublasHandle_t &handle,float *dev_c, const float *dev_a, const float *dev_b, unsigned int WA, unsigned int HA, unsigned int WB,
                           unsigned int HB);
}
#endif //PROJECT_MUL_CUBLAS_H
