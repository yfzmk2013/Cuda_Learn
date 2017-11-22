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
#include "matrixmul.h"

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
        //printf("%f,%f\n", _C[i], _D[i]);

        if (abs(_C[i] - _D[i]) >= 0.00000000000000000000000001) {
            isRight = false;
            printf("%d,%d,%f,%f\n", size, i, _C[i], _D[i]);
            //break;
        }
    }

    return isRight;
}

int main() {
    const int width_A = 128;
    const int height_A = 128;
    const int width_B = 128;
    const int height_B = 128;

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



    //产生随机数生成器
    srand((unsigned) time(0));

    randomInit(B, height_B * width_B);
    randomInit(A, height_A * width_A);

    //printMatrix(B, width_B, height_B);
    //printMatrix(A, width_A, height_A);

    //clock_t time_used;

    //CPU 计算
    //double Time = (double) cvGetTickCount();
    //clock_t start = clock();

//    MatrixMulCPU(C, A, B, width_A, height_A, width_B, height_B);
////    Time = (double) cvGetTickCount() - Time;
////    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));//
//
//    time_used = clock() - start;
//    printf("(CPU) time:%ld\n",  time_used);

    //GPU
    Type m_Mode = Mode1;

    //Time = (double) cvGetTickCount();
    //start = clock();
    double Time = (double) cvGetTickCount();

    cudaError_t cudaStatus = addWithCuda(D, A, B, width_A, height_A, width_B, height_B, m_Mode);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!\n");
        return 1;
    }
    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));


    //time_used = clock() - start;
    //printf("(GPU1) time:%ld\n", time_used);

//    Time = (double) cvGetTickCount() - Time;
//    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));//


    m_Mode = Mode2;
    Time = (double) cvGetTickCount();
    cudaStatus = addWithCuda(E, A, B, width_A, height_A, width_B, height_B, m_Mode);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!\n");
        return 1;
    }


//    time_used = clock() - start;
//    printf("(GPU2) time:%ld\n", time_used);

    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));


    //m_Mode = Mode3;

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

    Time = (double) cvGetTickCount();
//    cudaStatus = addWithCuda(F, A, B, width_A, height_A, width_B, height_B, m_Mode);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!\n");
//        return 1;
//    }

    cublasStatus = addWithCuda2(handle, F, A, B, width_A, height_A, width_B, height_B);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "addWithCuda failmaed!\n");
        return -1;
    }

//    time_used = clock() - start;
//    printf("(GPU2) time:%ld\n", time_used);

    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));

//    for (int i = 0; i <100 ; ++i) {
//        printf("%f,%f\n",D[i],F[i]);
//    }

    if (!CheckAnswer(D, F, height_A * width_B) /*&& !CheckAnswer(D, F, height_A * width_B)*/)
        printf("The answer is wrong!\n");
    else printf("The answer is right!\n");

    //检查GPU, CPU 计算的结果是否相同
//    if (!CheckAnswer(C, D, height_A * width_B) && !CheckAnswer(C, E, height_A * width_B))
//        printf("The answer is wrong!\n");
//    else printf("The answer is right!\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    //printMatrix(F,100,100);

    return 0;
}
