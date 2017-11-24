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


/**

 const int WA = 3;
    const int HA = 2;
    const int WB = 3;
    const int HB = 4;

    float A[WA * HA] = {1, 2, 3,
                        4, 5, 6};

    float B[WB * HB] = {1, 3, 9,
                        2, 1, 1,
                        3, 2, 5,
                        0, 2, 8
    };

 */


cublasStatus_t
addWithCuda5(const cublasHandle_t &handle, float *c, const float *a, const float *b, unsigned int WA, unsigned int HA,
             unsigned int WB,
             unsigned int HB) {

    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    printf("aaaaaaaaaaa!\n");


    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;


    const int WC = HB;
    const int HC = HA;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, WC * HC * sizeof(float));
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

    //printf("aaaaaaaaaaa!\n");

    int m = HB;
    int n = HA;
    int k = WB;
    int lda = WB;
    int ldb = WA;
    int ldc = WC;
    printf("%d,%d,%d,%d,%d,%d\n", m, n, k, lda, ldb, ldc);
    clock_t start = clock();


    cublasStatus = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dev_b, ldb, dev_a, lda, &beta, dev_c,
                               ldc);

    clock_t time_used = clock() - start;
    printf("(GPU31) time:%ld\n", time_used);

    cudaThreadSynchronize();

    //printf("aaaaaaaaaaa!\n");
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLASdddddd\n");

        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            printf("CUBLAS 对象实例化出错\n");
        }
        //return;
    }


    cudaThreadSynchronize();


    cudaThreadSynchronize();
    cublasGetVector(HC * WC, sizeof(float), dev_c, 1, c, 1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;


}

template<typename T>
void gpu_memory_alloc(size_t len, T *&ptr) {
    cudaMalloc(&ptr, sizeof(T) * len);
}

//#define CHECK_EQ(val1, val2) ((val1)==(val2))
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, > , val1, val2)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    std::cout<< " log:" << cudaGetErrorString(error)<<std::endl; \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 64;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


template<typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype *data_im,
                                  const int height, const int width, const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  Dtype *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        Dtype *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const Dtype *data_im_ptr = data_im;
        data_im_ptr += (c_im * height + h_offset) * width + w_offset;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                *data_col_ptr =
                        (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                        data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

template<typename Dtype>
void im2col_gpu(const Dtype *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype *data_col) {

    printf("XXXXXXXXXXXXXXXX\n");
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel<Dtype> << < CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS >> > (
                    num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
                            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                            width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}


void im2col2(const float *data_im, const int channels, int height, int width, const int kszie,
             const int pad, const int stride,
             float *data_col, int *h_out, int *w_out) {

    cudaError_t cudaStatus;
    float *dev_a = 0;
    float *dev_b = 0;


    int height_col = (height + 2 * pad -
                      ((kszie - 1) + 1)) / stride + 1;
    int width_col = (width + 2 * pad -
                     ((kszie - 1) + 1)) / stride + 1;

    *h_out = height_col;
    *w_out = width_col;

    cudaStatus = cudaMalloc((void **) &dev_a, channels * height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        // Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_b, channels * kszie * kszie * height_col * width_col * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        //goto Error;
    }

    cudaMemcpy(dev_a, data_im, sizeof(float) * channels * height * width, cudaMemcpyHostToDevice);


    printf("%d,%d,%d,%d,%d,%d\n", channels, height, width, kszie, pad, stride);

    im2col_gpu<float>(dev_a, channels, height, width, kszie, kszie, pad, pad, stride, stride, 1, 1, dev_b);

    cudaMemcpy(data_col, dev_b, channels * kszie * kszie * height_col * width_col * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);




//    *h_out = height_col;
//    *w_out = width_col;
//    float *pcol_1, *pcol_2;
//    float *pimg_1;
//
//    int c, h, w, k1, k2;
//    for (c = 0; c < channels; ++c) {
//        pcol_1 = data_col + c * kszie * kszie;
//        pimg_1 = (float *) (data_im + width * height * c);
//        int a = 0;
//        for (h = -pad; h < height_col * stride - pad; h += stride) {
//            for (w = -pad; w < width_col * stride - pad; w += stride) {
//
//                pcol_2 = pcol_1 + a * channels * kszie * kszie;
//                a++;
//                for (k1 = 0; k1 < kszie; ++k1) {
//                    for (k2 = 0; k2 < kszie; ++k2) {
//                        if (h + k1 < 0 || w + k2 < 0 || h + k1 > height - 1 || w + k2 > width - 1) {
//                            *pcol_2++ = 0;
//                        } else {
//                            *pcol_2++ = *(pimg_1 + (h + k1) * width + (w + k2));
//                        }
//                    }
//                }
//            }
//        }
//    }
}


void run22(const cublasHandle_t &handle, const cudaStream_t &stream, float *a, float *b, float *c) {


    float *d_a, *d_b, *d_c;


    gpu_memory_alloc<float>(6, d_a);
    gpu_memory_alloc<float>(8, d_b);
    gpu_memory_alloc<float>(12, d_c);

    cudaMemcpy(d_a, a, sizeof(float) * 6, cudaMemcpyDefault);
    cudaMemcpy(d_b, b, sizeof(float) * 8, cudaMemcpyDefault);
    float alph = 1.0f;
    float beta = 0.0f;

    /// a(3*2)    b(2 *4 )
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 3, 2, &alph, d_b, 4, d_a, 2, &beta, d_c, 4);
    cudaMemcpyAsync(c, d_c, 12 * sizeof(float), cudaMemcpyDefault, stream);

    cudaStreamSynchronize(stream);
    printf("aaaaaaaaaaaaaaaaaaaaaa!!!\n");


}

cublasStatus_t
addWithCuda6(const cublasHandle_t &handle, const float *a, const float *b, const int WA, const int HA,
             const int WB,
             const int HB, float *c) {
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;

    printf("aaaaaaaaaaa!\n");


    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;


    const int WC = WB;
    const int HC = HA;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, WC * HC * sizeof(float));
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

    //printf("aaaaaaaaaaa!\n");

    int m = WB;
    int n = HA;
    int k = WA;
    int lda = WA;
    int ldb = WB;
    int ldc = WB;
    printf("%d,%d,%d,%d,%d,%d\n", m, n, k, lda, ldb, ldc);
    clock_t start = clock();


    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_b, ldb, dev_a, lda, &beta, dev_c,
                               ldc);

    clock_t time_used = clock() - start;
    printf("(GPU31) time:%ld\n", time_used);

    cudaThreadSynchronize();

    //printf("aaaaaaaaaaa!\n");
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLASdddddd\n");

        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            printf("CUBLAS 对象实例化出错\n");
        }
        //return;
    }


    cudaThreadSynchronize();


    cudaThreadSynchronize();
    cublasGetVector(HC * WC, sizeof(float), dev_c, 1, c, 1);
    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cublasStatus;
}
