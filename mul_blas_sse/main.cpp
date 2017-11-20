//
// Created by yanhao on 17-11-20.
//

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "mul_blas_sse.h"
#include <opencv2/opencv.hpp>
using namespace cv;

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
        printf("%3.2f\t", m_Matrix[i]);
        if ((i+1) % W == 0 && i != 0) printf("\n");
    }
    printf("\n");
}

int main() {

    const int width_A = 2048;
    const int height_A = 2048;
    const int width_B = 2048;
    const int height_B = 2048;

    float *B = (float *) calloc(height_B * width_B, sizeof(float));
    float *A = (float *) calloc(height_A * width_A, sizeof(float));
    float *C = (float *) calloc(height_A * width_B, sizeof(float));
    float *D = (float *) calloc(height_A * width_B, sizeof(float));
    float *E = (float *) calloc(height_A * width_B, sizeof(float));

    memset(A, 0.0, sizeof(float) * height_A * width_A);
    memset(B, 0.0, sizeof(float) * height_B * width_B);
    memset(C, 0.0, sizeof(float) * height_A * width_B);
    memset(D, 0.0, sizeof(float) * height_A * width_B);
    memset(E, 0.0, sizeof(float) * height_A * width_B);

    randomInit(B, height_B * width_B);
    randomInit(A, height_A * width_A);


    double Time = (double) cvGetTickCount();
    run_gemm(A, B, C, 2048, 2048, 2048, 1);
    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));
    //printMatrix(C, 2048, 2048);

    printf("\n");

    Time = (double) cvGetTickCount();
    run_gemm(A, B, C, 2048, 2048, 2048, 0);
    Time = (double) cvGetTickCount() - Time;
    printf("run time = %gms\n", Time / (cvGetTickFrequency() * 1000));

    //printMatrix(C, 2048, 2048);

    //产生随机数生成器
    srand(clock());

    printf("hello,world!\n");

}