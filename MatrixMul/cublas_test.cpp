//
// Created by yanhao on 17-11-20.
//
#include <cublas_v2.h> //cuda自带库函数
//#include "helper_cuda.h"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

inline void checkCudaErrors(cudaError err)
 {
     if(cudaSuccess != err)
     {
         fprintf(stderr,  ": CUDA Runtime API error %d: %s.\n",(int)err, cudaGetErrorString( err ) );
         return ;
     }
}

//给初始的矩阵一个随机值
void randomInit(float *_data, int _size) {
    for (int i = 0; i < _size; ++i) {
        _data[i] = rand() / (float) RAND_MAX;
        //printf("%f\n",_data[i]);
    }
}

int main(void) {
#if 0
    float alpha = 1.0;
    float beta = 0.0;
    float h_A[6] = {1, 1, 2, 2, 3, 3};
    float h_B[2] = {1, 1};
    float h_C[3];
    float *d_a, *d_b, *d_c;

    checkCudaErrors(cudaMalloc((void **) &d_a, 6 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_b, 2 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_c, 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_a, &h_A, 6 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, &h_B, 2 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_c, 0, 3 * sizeof(float)));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 3, 2, &alpha, d_b, 1, d_a, 2, &beta, d_c, 1);
    checkCudaErrors(cudaMemcpy(h_C, d_c, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 3; i++) {
        printf("%f\n", h_C[i]);
    }
    printf("\n");

#endif

#if 1

    const int width_A = 128;
    const int height_A = 128;
    const int width_B = 128;
    const int height_B = 128;

    float *B = (float *) calloc( height_B * width_B,sizeof(float));
    float *A = (float *) calloc( height_A * width_A,sizeof(float));
    float *C = (float *) calloc(height_A * width_B,sizeof(float));
//    float *D = (float *) calloc(height_A * width_B,sizeof(float));
//    float *E = (float *) calloc(height_A * width_B,sizeof(float));

    memset(A, 0.0, sizeof(float) * height_A * width_A);
    memset(B, 0.0, sizeof(float) * height_B * width_B);
    memset(C, 0.0, sizeof(float) * height_A * width_B);
//    memset(D, 0.0, sizeof(float) * height_A * width_B);
//    memset(E, 0.0, sizeof(float) * height_A * width_B);


    srand((unsigned) clock());

    randomInit(B, height_B * width_B);
    randomInit(A, height_A * width_A);
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void **) &d_a, height_A * width_A * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_b, height_B * width_B * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_c, height_B * width_B * sizeof(float)));



    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        //goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMemcpy((void **) &d_a, A, height_A * width_A * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    //cudaMemcpy(d_a, &A, height_A * width_A * sizeof(float), cudaMemcpyHostToDevice);
//
    //checkCudaErrors(cudaMemcpy(d_a, &A, height_A * width_A * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(d_b, &B, height_B * width_B * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemset(d_c, 0, height_B * width_B * sizeof(float)));
//
//    cublasHandle_t handle;
//    cublasCreate(&handle);
//    float alpha = 1.0;
//    float beta = 0.0;
//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,128, 128, 128, &alpha, d_b, 128, d_a,128, &beta, d_c, 128);
//    checkCudaErrors(cudaMemcpy(C, d_c, 128 *128* sizeof(float), cudaMemcpyDeviceToHost));



    //产生随机数生成器




#endif
    return 0;
}
