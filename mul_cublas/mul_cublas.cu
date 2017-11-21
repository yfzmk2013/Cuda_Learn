//
// Created by yanhao on 17-11-21.
//


#include <stdio.h>
#include <stdlib.h>
//#include "utils.h"
#include <iostream>


//#include "helper_cuda.h"
#include <stdio.h>
#include <cuda_runtime_api.h>

#include "mul_cublas.h"
// Helper function for using CUDA to add vectors in parallel.
cublasStatus_t addWithCuda(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                        unsigned int HB) {

    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_a, HA * WA * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_b, HB * WB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cublasSetVector(HA*WA,sizeof(float),a,1,dev_a,1);
    cublasSetVector(HB*WB,sizeof(float),b,1,dev_b,1);
    // 同步函数
    cudaThreadSynchronize();

    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c, HA);

    cudaThreadSynchronize();
    cublasGetVector(HA*WB,sizeof(float),c,1,dev_c,1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;
}

cublasStatus_t addWithCuda2(const cublasHandle_t &handle,float *dev_c, const float *dev_a, const float *dev_b, unsigned int WA, unsigned int HA, unsigned int WB,
                            unsigned int HB){

    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c, HA);


}