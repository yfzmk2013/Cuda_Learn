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
//cublasStatus_t
//addWithCuda(const cublasHandle_t &handle, float *c, const float *a, const float *b, unsigned int WA, unsigned int HA,
//            unsigned int WB,
//            unsigned int HB) {
//
//    float *dev_a = 0;
//    float *dev_b = 0;
//    float *dev_c = 0;
//    cudaError_t cudaStatus;
//    cublasStatus_t cublasStatus;
//
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        // Error;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_a, HA * WA * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        //goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_b, HB * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        //goto Error;
//    }
//
//    cublasSetVector(HA * WA, sizeof(float), a, 1, dev_a, 1);
//    cublasSetVector(HB * WB, sizeof(float), b, 1, dev_b, 1);
//    // 同步函数
//    cudaThreadSynchronize();
//
//    float alpha = 1.0;
//    float beta = 0.0;
//    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c,
//                               HA);
//
//    cudaThreadSynchronize();
//    cublasGetVector(HA * WB, sizeof(float), c, 1, dev_c, 1);
//    //Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    return cublasStatus;
//}

//cublasStatus_t
//addWithCuda(const cublasHandle_t &handle, float *c, const float *a, const float *b, unsigned int WA, unsigned int HA,
//            unsigned int WB,
//            unsigned int HB) {
//
//    float *dev_a = 0;
//    float *dev_b = 0;
//    float *dev_c = 0;
//    cudaError_t cudaStatus;
//    cublasStatus_t cublasStatus;
//
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        // Error;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_a, HA * WA * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        //goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_b, HB * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        printf( "cudaMalloc failed!");
//        //goto Error;
//    }
//
//    cublasSetVector(HA * WA, sizeof(float), a, 1, dev_a, 1);
//    cublasSetVector(HB * WB, sizeof(float), b, 1, dev_b, 1);
//    // 同步函数
//    cudaThreadSynchronize();
//
//    float alpha = 1.0;
//    float beta = 0.0;
//    clock_t start = clock();
//
//    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c,
//                               HA);
//
//
//    clock_t time_used = clock() - start;
//    printf("(GPU31) time:%ld\n", time_used);
//    cudaThreadSynchronize();
//    cublasGetVector(HA * WB, sizeof(float), c, 1, dev_c, 1);
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//            printf("%f\n", c[i * 2 + j]);
//        }
//    }
//    //Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    return cublasStatus;
//}


//addWithCuda2(const cublasHandle_t &handle, float *dev_c, const float *dev_a, const float *dev_b, unsigned int WA,
//             unsigned int HA, unsigned int WB,
//             unsigned int HB) {
//
//    float alpha = 1.0;
//    float beta = 0.0;
//    cublasStatus_t cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WA, HA, WB, &alpha, dev_b, HA, dev_a,
//                                              HA, &beta, dev_c, HA);
//
//
//}

cublasStatus_t addWithCuda2(const cublasHandle_t &handle,float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
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

    cublasSetVector(HA * WA, sizeof(float), a, 1, dev_a, 1);
    cublasSetVector(HB * WB, sizeof(float), b, 1, dev_b, 1);
    // 同步函数
    cudaThreadSynchronize();

    float alpha = 1.0;
    float beta = 0.0;
    clock_t start = clock();

    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c,
                               HA);

    cudaThreadSynchronize();

    clock_t time_used = clock() - start;
    printf("(GPU31) time:%ld\n", time_used);
    cudaThreadSynchronize();
    cublasGetVector(HA * WB, sizeof(float), dev_c, 1,c, 1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;
}

template< typename T>
void gpu_memory_alloc(size_t len, T * &ptr)
{
    cudaMalloc(&ptr, sizeof(T) * len);
}
void run(const cublasHandle_t &handle,const cudaStream_t&stream,float *a,float*b,float *c)
{


    float *d_a , *d_b, *d_c;



    gpu_memory_alloc<float>(6, d_a);
    gpu_memory_alloc<float>(8, d_b);
    gpu_memory_alloc<float>(12, d_c);

    cudaMemcpy(d_a, a, sizeof(float)* 6, cudaMemcpyDefault);
    cudaMemcpy(d_b, b, sizeof(float)* 8, cudaMemcpyDefault);
    float alph = 1.0f;
    float beta = 0.0f;

    /// a(3*2)    b(2 *4 )
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 3, 2, &alph, d_b, 4, d_a, 2 ,&beta, d_c, 4 );
    cudaMemcpyAsync(c, d_c, 12 * sizeof(float), cudaMemcpyDefault, stream);

    cudaStreamSynchronize(stream);
    printf("aaaaaaaaaaaaaaaaaaaaaa!!!\n");


}