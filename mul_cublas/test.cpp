//
// Created by yanhao on 17-11-20.
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <Windows.h>
#include <string.h>
#include <malloc.h>
#include "opencv2/opencv.hpp"
#include "device_functions.h"
#include "mul_cublas.h"
#include <cublas_v2.h> //cuda自带库函数

using namespace cv;



//cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
//                        unsigned int HB, Type mode);


//GPU version
void MatrixMulCPU(float *_C, const float *_A, const float *_B, int WA, int HA, int WB, int HB) {
    if (WA != HB) {
        printf("the matrix A and B cannot be multipled!");
        exit(0);
    }

    for (int i = 0; i < HA; ++i) {
        for (int j = 0; j < WB; ++j) {
            for (int k = 0; k < WA; ++k) {
                _C[i * WA + j] += _A[i * WA + k] * _B[k * WB + j];
            }
        }
    }
}

//给初始的矩阵一个随机值
void randomInit(float *_data, int _size) {
    for (int i = 0; i < _size; ++i) {
        _data[i] = rand() / (float) RAND_MAX;
        //printf("%f\n",_data[i]);
    }
}

//print the matrix
void printMatrix(float *m_Matrix, int W, int H) {
    for (int i = 0; i < W * H; ++i) {
        printf("%2.1f ", m_Matrix[i]);
        if (i % W == 0 && i != 0) printf("\n");
    }
    printf("\n");
}

bool CheckAnswer(const float *_C, const float *_D, unsigned int size) {
    bool isRight = true;
    for (int i = 0; i < size && isRight == true; ++i) {
        if (abs(_C[i] - _D[i]) >= 0.0000000000001) {
            isRight = false;
            printf("%d,%d,%f,%f\n", size, i, _C[i], _D[i]);
            //break;
        }
    }

    return isRight;
}

int main() {


    const int width_A = 4096;
    const int height_A = 4096;
    const int width_B = 4096;
    const int height_B = 4096;

    float *B = (float *) calloc(height_B * width_B, sizeof(float));
    float *A = (float *) calloc(height_A * width_A, sizeof(float));
    float *C = (float *) calloc(height_A * width_B, sizeof(float));
    float *D = (float *) calloc(height_A * width_B, sizeof(float));
    float *E = (float *) calloc(height_A * width_B, sizeof(float));
    float *F = (float *) calloc(height_A * width_B, sizeof(float));


    memset(A, 0.0, sizeof(float) * height_A * width_A);
    memset(B, 0.0, sizeof(float) * height_B * width_B);
    memset(C, 0.0, sizeof(float) * height_A * width_B);
    memset(D, 0.0, sizeof(float) * height_A * width_B);
    memset(E, 0.0, sizeof(float) * height_A * width_B);
    memset(F, 0.0, sizeof(float) * height_A * width_B);


    // cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        return -1;
//    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            //cout << "CUBLAS 对象实例化出错" << endl;
        }
        getchar();
        printf("hello,is r\n");
        return -1;
    }

    //产生随机数生成器
    srand((unsigned) clock());
    randomInit(B, height_B * width_B);
    randomInit(A, height_A * width_A);


#if 1

    double Time = (double) cvGetTickCount();
    Time = (double) cvGetTickCount();
    cublasStatus = addWithCuda(handle, F, A, B, width_A, height_A, width_B, height_B);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "addWithCuda failmaed!\n");
        return -1;
    }
    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));


    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);

    // 释放 CUBLAS 库对象
    cublasDestroy (handle);
#endif


#if 0

    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
    //cublasStatus_t cublasStatus;
    int HA = height_A;
    int WB = height_B;
    int WA = width_A;
    int HB = height_B;


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

    cublasSetVector(HA * WA, sizeof(float), A, 1, dev_a, 1);
    cublasSetVector(HB * WB, sizeof(float), B, 1, dev_b, 1);
    // 同步函数
    cudaThreadSynchronize();

    float alpha = 1.0;
    float beta = 0.0;
    double Time = (double) cvGetTickCount();
    //cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c, HA);
    cublasStatus = addWithCuda2(handle, dev_c, dev_a, dev_b, WA, HA, WB, HB);

    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));

    cudaThreadSynchronize();
    cublasGetVector(HA * WB, sizeof(float), C, 1, dev_c, 1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;

#endif
    return 0;
}
