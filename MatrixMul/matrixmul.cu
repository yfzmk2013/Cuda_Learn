#include <stdio.h>
#include <stdlib.h>
//#include "utils.h"
#include <iostream>
#include "matrixmul.h"


#include <cublas_v2.h> //cuda自带库函数
//#include "helper_cuda.h"
#include <stdio.h>
#include <cuda_runtime_api.h>


// ======== define area ========
#define DATA_SIZE 1048576 // 1M

// ======== global area ========
int data[DATA_SIZE];

__global__ void MatrixMulGPU_1(float *c, const float *a, const float *b, unsigned int WA, unsigned int WB) {
    float sum = 0;
    //找出该线程所在的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //线程Thread(row, col)负责计算C(row, col)
    for (int i = 0; i < WB; ++i) {
        sum += a[row * WA + i] * b[i * WB + col];
    }

    c[row * WB + col] = sum;
}

//template <int BS,int MA,int MB>
//__global__ void MatrixMulGPU_3(int matrixRowsA,const int matrixColsARowsB,int matrixColsB,
//const float4*)

template<int BLOCK_SIZE>
__global__ void MatrixMulGPU_2(float *c, const float *a, const float *b, unsigned int WA, unsigned int WB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + WA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int i = aBegin, j = bBegin;
         i <= aEnd;
         i += aStep, j += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = a[i + WA * ty + tx];
        Bs[ty][tx] = b[j + WB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        //__syncthreads();
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int k = WB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[k + WB * ty + tx] = Csub;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int WA, unsigned int HA, unsigned int WB,
                        unsigned int HB, Type mode) {

    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        //goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
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

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, HA * WA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, HB * WB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    //为每一个C[i][j]设置一个线程进行计算
    int block_size = 16;

    dim3 Threads(block_size, block_size);
    dim3 Blocks(WB / block_size, HA / block_size);

    // Launch a kernel on the GPU with one thread for each element.
    if (mode == Mode1) {
        clock_t time_used;
        clock_t start = clock();

        MatrixMulGPU_1 << < Blocks, Threads >> > (dev_c, dev_a, dev_b, WA, WB);
        time_used = clock() - start;
       // printf("(GPU1) time:%ld\n", time_used);
    }

    if (mode == Mode2) {

        clock_t time_used;
        clock_t start = clock();
        MatrixMulGPU_2<16> << < Blocks, Threads >> > (dev_c, dev_a, dev_b, WA, WB);
        time_used = clock() - start;
        //printf("(GPU2) time:%ld\n", time_used);
    }

    if(mode==Mode3){
        clock_t start = clock();

        cublasHandle_t handle;
        cublasCreate(&handle);
        clock_t time_used = clock() - start;
        printf("(GPU31) time:%ld\n", time_used);

        float alpha = 1.0;
        float beta = 0.0;
        fprintf(stderr, "aaaaaaaaaaaaaa!");
        start = clock();

        //clock_t time_used;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c, HA);


        time_used = clock() - start;
        printf("(GPU3) time:%ld\n", time_used);

//        if (cudaStatus != cudaSuccess) {
//            fprintf(stderr, "cudaMemcpy failed!");
//            //goto Error;
//        }


    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        //goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, HA * WB * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
    }

    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

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
    cublasGetVector(HA * HB, sizeof(float), dev_c, 1,c, 1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;
}