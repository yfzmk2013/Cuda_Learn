//
// Created by yanhao on 17-11-23.
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

#define W_B_LEN 20
#define W_B_Data_Dim 4
#define FC_W_H 21504
#define FC_W_W 512

#define SAFE_FREE(p)      {if((p) != NULL) {free(p); (p) = NULL;}}
#define SAFE_CLOSE(fp)    {if((fp) != NULL) {fclose((fp)); (fp) = NULL;}}

typedef struct _MODEL_LEN {
    int k1;
    int k2;
    int in_len;
    int out_len;
} MODEL_LEN;

typedef struct _CNN_Model {
    MODEL_LEN *model_len;
    float **CNN_W;
    float **CNN_B;
    float **CNN_Prelu;
    float *CNN_fc_w;
    float *CNN_fc_b;
} CNN_Model;


typedef struct _CNN_Data {
    float *data;
    float *data1;
    float *dstdata;
    float *data_cp;
} CNN_Data;

void checkCudaErrors(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << std::endl;
        exit(-1);
//        if( abort )
//            exit( code );
    }
}

int init(CNN_Model &cnn_model) {

    FILE *fp_cnn_len = fopen("/home/yanhao/tmpCNN/model_300.bin", "rb");
    FILE *fp_cnn_w = fopen("/home/yanhao/tmpCNN/model_301.bin", "rb");
    FILE *fp_cnn_b = fopen("/home/yanhao/tmpCNN/model_302.bin", "rb");
    FILE *fp_cnn_prelu = fopen("/home/yanhao/tmpCNN/model_303.bin", "rb");
    FILE *fp_cnn_fc_w = fopen("/home/yanhao/tmpCNN/model_304.bin", "rb");
    FILE *fp_cnn_fc_b = fopen("/home/yanhao/tmpCNN/model_305.bin", "rb");

    if (!fp_cnn_len || !fp_cnn_w || !fp_cnn_b || !fp_cnn_prelu || !fp_cnn_fc_w || !fp_cnn_fc_b) {
        printf("open model file error!\n");
        return -1;
    }

    int len[W_B_LEN * W_B_Data_Dim];
    MODEL_LEN model_len[W_B_LEN];
    fread(len, sizeof(int), W_B_LEN * W_B_Data_Dim, fp_cnn_len);

    for (int i = 0; i < W_B_LEN; ++i) {
        model_len[i].k1 = len[W_B_Data_Dim * i];
        model_len[i].k2 = len[W_B_Data_Dim * i + 1];
        model_len[i].in_len = len[W_B_Data_Dim * i + 2];
        model_len[i].out_len = len[W_B_Data_Dim * i + 3];
    }

    cnn_model.model_len = (MODEL_LEN *) malloc(W_B_LEN * sizeof(MODEL_LEN));
    cnn_model.CNN_W = (float **) malloc(W_B_LEN * sizeof(float *));
    cnn_model.CNN_B = (float **) malloc(W_B_LEN * sizeof(float *));
    cnn_model.CNN_Prelu = (float **) malloc(W_B_LEN * sizeof(float *));
    cnn_model.CNN_fc_w = (float *) malloc(FC_W_H * FC_W_W * sizeof(float));
    cnn_model.CNN_fc_b = (float *) malloc(FC_W_W * sizeof(float));

    if (!cnn_model.model_len || !cnn_model.CNN_W || !cnn_model.CNN_B
        || !cnn_model.CNN_Prelu || !cnn_model.CNN_fc_w || !cnn_model.CNN_fc_b) {
        printf("molloc error!\n");
        return -1;
    }

    fread(cnn_model.CNN_fc_w, sizeof(float), FC_W_H * FC_W_W, fp_cnn_fc_w);
    fread(cnn_model.CNN_fc_b, sizeof(float), FC_W_W, fp_cnn_fc_b);


    for (int k = 0; k < W_B_LEN; ++k) {
        int k1 = model_len[k].k1;
        int k2 = model_len[k].k2;
        int in_len = model_len[k].in_len;
        int out_len = model_len[k].out_len;
        cnn_model.CNN_W[k] = (float *) malloc(sizeof(float) * k1 * k2 * in_len * out_len);
        cnn_model.CNN_B[k] = (float *) malloc(sizeof(float) * 1 * out_len);
        cnn_model.CNN_Prelu[k] = (float *) malloc(sizeof(float) * 1 * out_len);

        if (!cnn_model.CNN_W[k] || !cnn_model.CNN_B[k] || !cnn_model.CNN_Prelu[k]) {
            printf("molloc error!\n");
            return -1;
        }

        fread(cnn_model.CNN_W[k], sizeof(float), k1 * k2 * in_len * out_len, fp_cnn_w);
        fread(cnn_model.CNN_B[k], sizeof(float), 1 * out_len, fp_cnn_b);
        fread(cnn_model.CNN_Prelu[k], sizeof(float), 1 * out_len, fp_cnn_prelu);
    }


    for (int j = 0; j < W_B_LEN; ++j) {
        printf("%d,%d,%d,%d\n", model_len[j].k1, model_len[j].k2, model_len[j].in_len, model_len[j].out_len);
    }

    for (int l = 0; l < W_B_LEN; ++l) {
        cnn_model.model_len[l].k1 = model_len[l].k1;
        cnn_model.model_len[l].k2 = model_len[l].k2;
        cnn_model.model_len[l].in_len = model_len[l].in_len;
        cnn_model.model_len[l].out_len = model_len[l].out_len;
    }

    SAFE_CLOSE(fp_cnn_len);
    SAFE_CLOSE(fp_cnn_w);
    SAFE_CLOSE(fp_cnn_b);
    SAFE_CLOSE(fp_cnn_prelu);
    SAFE_CLOSE(fp_cnn_fc_w);
    SAFE_CLOSE(fp_cnn_fc_b);
    printf("init ok!\n");
    return 0;
}

static void rgb2Mat(IplImage *img, unsigned char *mat) {
    int i, j, offset;

    for (i = 0; i < img->height; i++) {
        for (j = 0; j < img->width; j++) {
            for (int k = 0; k < 3; ++k) {
                offset = (i * img->widthStep + j * 3 + k);
                mat[(i * img->width + j) * 3 + k] = *(img->imageData + offset);
            }
        }
    }
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
//  do {\
//    cudaError_t error = condition; \
//    //std::cout<< " log:" << cudaGetErrorString(error)<<std::endl; \
//  } while (0)

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


__global__ void im2col_gpu_kernel(const int n, const float *data_im,
                                  const int height, const int width, const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  float *data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        const int h_index = index / width_col;
        const int h_col = h_index % height_col;
        const int w_col = index % width_col;
        const int c_im = h_index / height_col;
        const int c_col = c_im * kernel_h * kernel_w;
        const int h_offset = h_col * stride_h - pad_h;
        const int w_offset = w_col * stride_w - pad_w;
        float *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
        const float *data_im_ptr = data_im;
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


void im2col_gpu(const float *data_im, const int channels,
                const int height, const int width, const int kernel,
                const int pad,
                const int stride,
                float *data_col, int *h_out, int *w_out) {

    const int kernel_h = kernel;
    const int kernel_w = kernel;
    const int dilation_h = 1;
    const int dilation_w = 1;
    const int pad_h = pad;
    const int pad_w = pad;
    const int stride_h = stride;
    const int stride_w = stride;

    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
                      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
                     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    *h_out = height_col;
    *w_out = width_col;
    int num_kernels = channels * height_col * width_col;
    // NOLINT_NEXT_LINE(whitespace/operators)
    im2col_gpu_kernel << < CAFFE_GET_BLOCKS(num_kernels),
            CAFFE_CUDA_NUM_THREADS >> > (
                    num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
                            pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
                            width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}


__global__ void PReLUForward(const int n, const int channels, const int dim,
                             const float *in, float *out, const float *b_data, const float *slope_data,
                             const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        out[index] = (in[index] + b_data[c]) > 0 ? (in[index] + b_data[c]) : (in[index] + b_data[c]) * slope_data[c];
    }
}

__global__ void ADD_GPU(const float *src1, const float *src2,const int n, float *dst) {
    CUDA_KERNEL_LOOP(index, n) {
        //int c = (index / dim) % channels / div_factor;
        dst[index] =src1[index] + src2[index];
    }
}

void Bi_D_gpu(float *src, float *dst, const int dim, const int channels, const float *b_data, const float *slope_data) {
//const Dtype* bottom_data = bottom[0]->gpu_data();
//Dtype* top_data = top[0]->mutable_gpu_data();
//const int count = bottom[0]->count();
//const int dim = bottom[0]->count(2);
//const int channels = bottom[0]->channels();
//const Dtype* slope_data = this->blobs_[0]->gpu_data();
//const int div_factor = channel_shared_ ? channels : 1;
//
//// For in-place computation
//if (top[0] == bottom[0]) {
//caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
//}
    const int count = dim * channels;
// NOLINT_NEXT_LINE(whitespace/operators)
    PReLUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> > (
            count, channels, dim, src, dst, b_data, slope_data, 1);
    CUDA_POST_KERNEL_CHECK;
}
void ADD_G(const float *src1,const float*src2,const int count, float*dst){
    ADD_GPU<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(src1,src2,count,dst);

    CUDA_POST_KERNEL_CHECK;
}

//cublasStatus_t
//addWithCuda6(const cublasHandle_t &handle, const float *dev_a, const float *dev_b, const int WA, const int HA,
//             const int WB,
//             const int HB, float *dev_c) {
////    if(WA!=HB || WA<=0 || WB <=0 ||HA <=0 || HB <=0 || !a || !b || !c){
////        return CUBLAS_STATUS_INTERNAL_ERROR;
////    }
////
////
////    float *dev_a = 0;
////    float *dev_b = 0;
////    float *dev_c = 0;
//    cudaError_t cudaStatus;
//    cublasStatus_t cublasStatus;
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        return CUBLAS_STATUS_INTERNAL_ERROR;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_a, HA * WA * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        return CUBLAS_STATUS_INTERNAL_ERROR;
//    }
//
//    cudaStatus = cudaMalloc((void **) &dev_b, HB * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        return CUBLAS_STATUS_INTERNAL_ERROR;
//    }
//
//    cublasSetVector(HA * WA, sizeof(float), a, 1, dev_a, 1);
//    cublasSetVector(HB * WB, sizeof(float), b, 1, dev_b, 1);
//
//    // 同步函数
//    cudaThreadSynchronize();
//    float alpha = 1.0;
//    float beta = 0.0;
//    clock_t start = clock();
//
//    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WA, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c,
//                               HA);
//
//    cudaThreadSynchronize();
//
//    clock_t time_used = clock() - start;
//    printf("(GPU31) time:%ld\n", time_used);
//    cudaThreadSynchronize();
//    cublasGetVector(HA * HB, sizeof(float), dev_c, 1, c, 1);
//    //Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    return cublasStatus;
//}


void change_zhizheng(float **data, float **dstdata, int &data_h, int &data_w, int &h_out, int &w_out) {
    float *tmp;
    tmp = *data;
    *data = *dstdata;
    *dstdata = tmp;
    data_h = h_out;
    data_w = w_out;
}


cublasStatus_t
run_cublasgemm(const cublasHandle_t &handle, const float *dev_a, const float *dev_b, float *dev_c, const int HA,
               const int WB,
               const int WA, int Mode = 0) {

//    float *dev_a = 0;
//    float *dev_b = 0;
//    float *dev_c = 0;

//
//
//    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;



    // 同步函数
    //cudaThreadSynchronize();

    float alpha = 1.0;
    float beta = 0.0;
    //printf("aaaaaaaaaaa!\n");

    int m = WB;
    int n = HA;
    int k = WA;
    int lda = WA;
    int ldb = WB;
    int ldc = WB;


    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dev_b, ldb, dev_a, lda, &beta, dev_c,
                               ldc);
    //cudaThreadSynchronize();
    //cudaThreadSynchronize();
    return cublasStatus;
}

void
run(const cublasHandle_t &handle, float *dev_data, int *dev_data_h, int *dev_data_w, float *dev_data1,
    float *dev_dstdata,
    const CNN_Model &dev_cnn_model, int ID, int pad,
    int stride, int *h_out, int *w_out) {

    const int Mode = 0;
    int kern = dev_cnn_model.model_len[ID].k1;
    int in_len = dev_cnn_model.model_len[ID].in_len;
    int out_len = dev_cnn_model.model_len[ID].out_len;
    //printf("aa\n");

    im2col_gpu(dev_data, in_len, *dev_data_h, *dev_data_w, kern, pad, stride, dev_data1, h_out, w_out);
//    float *b = (float *) malloc(sizeof(float) * 1000);
//    cudaMemcpy(b,dev_cnn_model.CNN_W[ID],sizeof(float) * 1000,cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < 1000; ++i) {
//
//        printf("%f,%f,%f\n",b[i],b[i],b[i]);
//
//    }

    //printf("aaa");
    float alpha = 1.0;
    float beta = 0.0;
    clock_t start = clock();

    cublasStatus_t cublasStatus;

    cublasStatus = run_cublasgemm(handle, dev_cnn_model.CNN_W[ID], dev_data1, dev_dstdata, out_len, (*h_out) * (*w_out),
                                  in_len * kern * kern, Mode);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            //cout << "CUBLAS 对象实例化出错" << endl;
        }
        printf("CUBLAS 对象实例化出错\n");
        //getchar();
        printf("hello,is r\n");
        exit(-1);
    }

    Bi_D_gpu(dev_dstdata,dev_dstdata,(*h_out) * (*w_out),dev_cnn_model.model_len[ID].out_len,
             dev_cnn_model.CNN_B[ID],dev_cnn_model.CNN_Prelu[ID]);







    //printf("%f,%f,%f,%d\n",c[0],c[1],c[2],ID);






    if(ID==190) {

        float *c= (float *) malloc((*h_out) * (*w_out)*dev_cnn_model.model_len[ID].out_len*sizeof(float));
//        float b = dev_cnn_model.CNN_B[ID][0];
        cudaError_t cudaStatus = cudaMemcpy(c,dev_dstdata,(*h_out) * (*w_out)*dev_cnn_model.model_len[ID].out_len*sizeof(float),cudaMemcpyDeviceToHost);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            exit(-1);
        }

        float *p_im_data = c;//dev_dstdata;
        for (int l = 0; l < dev_cnn_model.model_len[ID].out_len; ++l) {
            //float b = dev_cnn_model.CNN_B[ID][l];
            //float ai = dev_cnn_model.CNN_Prelu[ID][l];
            for (int k = 0; k < (*h_out) * (*w_out); ++k) {
                //float tmp = *p_im_data + b;
                //*p_im_data++ = (tmp > 0 ? tmp : (tmp * ai));
                printf("%d:%d:%f\n",ID,l*(*h_out) * (*w_out)+k, *p_im_data++);
            }
        }
    }





}

//__global__ void add_gpu(float *a,  float *b, float *c, int n) {
//    int i = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
//    if (i < n) {
//        c[i] = a[i] + b[i];
//    }
//}

void runFace(const cublasHandle_t &handle, float *dataIn, int w, int h, int c, const CNN_Model &dev_cnn_model,
             CNN_Data &dev_cnn_data, float *dataOut,
             int FeaLen) {

    int data_h = h;
    int data_w = w;
    int h_out;
    int w_out;
    float *data = dev_cnn_data.data;
    float *data1 = dev_cnn_data.data1;
    float *dstdata = dev_cnn_data.dstdata;
    float *data_cp = dev_cnn_data.data_cp;



    double Time = (double) cvGetTickCount();
    cudaMemcpy(data, dataIn, sizeof(float) * w * h * c, cudaMemcpyHostToDevice);

    //cudaThreadSynchronize();
    //printf("ccc\n");


    //c_1_1
    run(handle, data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 0, 1, 2, &h_out, &w_out);
//cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[0].out_len * h_out * w_out * sizeof(float));
    cudaMemcpy(data_cp,dstdata,dev_cnn_model.model_len[0].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //cudaThreadSynchronize();

    //c_1_2
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 1, 1, 1, &h_out, &w_out);
    //c_1_3
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 2, 1, 1, &h_out, &w_out);
    //rest_1_3
//    float *p = data_cp;
//    float *q = dstdata;
//    for (int m = 0; m < cnn_model.model_len[0].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }

    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[0].out_len * h_out * w_out,dstdata);
    //c_2_1
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 3, 1, 2, &h_out, &w_out);
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[3].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToHost);


    //c_2_2
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 4, 1, 1, &h_out, &w_out);
    //c_2_3
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 5, 1, 1, &h_out, &w_out);
    //res_2_3
    //cudaThreadSynchronize();

    /*{
        //res_2_3
        p = data_cp;
        q = dstdata;
        for (int m = 0; m < cnn_model.model_len[3].out_len * h_out * w_out; ++m) {
            *q = *q + (*p);
            q++;
            p++;
        }
        memcpy(data_cp, dstdata, cnn_model.model_len[3].out_len * h_out * w_out * sizeof(float));
        //c_2-4
        change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
        run(data, &data_h, &data_w, data1, dstdata, cnn_model, 6, 1, 1, &h_out, &w_out);
        //c_2_5
    }*/



    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[3].out_len * h_out * w_out,dstdata);
    //ADD_G(dstdata,data_cp,dev_cnn_model.model_len[0].out_len * h_out * w_out,dstdata);




    //cudaThreadSynchronize();


//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[3].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }

    //memcpy(data_cp, dstdata, cnn_model.model_len[3].out_len * h_out * w_out * sizeof(float));


    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[3].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //cudaThreadSynchronize();

    //c_2-4
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 6, 1, 1, &h_out, &w_out);
    //c_2_5
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 7, 1, 1, &h_out, &w_out);
    //res_2_5
    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[3].out_len * h_out * w_out,dstdata);

//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[3].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }
    //c_3_1
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 8, 1, 2, &h_out, &w_out);
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[8].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //c_3_2
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 9, 1, 1, &h_out, &w_out);
    //c_3_3
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 10, 1, 1, &h_out, &w_out);
    //res_3_3


    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[8].out_len * h_out * w_out,dstdata);

//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[8].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[10].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //c_3_4
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 11, 1, 1, &h_out, &w_out);
    //c_3_5
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 12, 1, 1, &h_out, &w_out);
    //res_3_5
    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[10].out_len * h_out * w_out,dstdata);


//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[10].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[12].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //c_3_6
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 13, 1, 1, &h_out, &w_out);
    //c_3_7
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 14, 1, 1, &h_out, &w_out);
    //res_3_7

    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[12].out_len * h_out * w_out,dstdata);

//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[12].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[14].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //c_3_8
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 15, 1, 1, &h_out, &w_out);
    //c_3_9
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 16, 1, 1, &h_out, &w_out);
    //res_3_9

    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[14].out_len * h_out * w_out,dstdata);

//    p = data_cp;
//    q = dstdata;
//    for (int m = 0; m < dev_cnn_model.model_len[14].out_len * h_out * w_out; ++m) {
//        *q = *q + (*p);
//        q++;
//        p++;
//    }
    //c_4_1
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 17, 1, 2, &h_out, &w_out);
    cudaMemcpy(data_cp, dstdata, dev_cnn_model.model_len[17].out_len * h_out * w_out * sizeof(float),cudaMemcpyDeviceToDevice);
    //c_4_2
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 18, 1, 1, &h_out, &w_out);
    //c_4_3
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);
    run(handle,data, &data_h, &data_w, data1, dstdata, dev_cnn_model, 19, 1, 1, &h_out, &w_out);
    //res_4_3

    ADD_G(dstdata,data_cp,dev_cnn_model.model_len[17].out_len * h_out * w_out,dstdata);


    //fc
    change_zhizheng(&data, &dstdata, data_h, data_w, h_out, w_out);


    cublasStatus_t cublasStatus;

    //Time = (double) cvGetTickCount();

    //run_gemm(cnn_model.CNN_fc_w, data, dstdata, FC_W_W, 1, FC_W_H, 1);
    cublasStatus = run_cublasgemm(handle, dev_cnn_model.CNN_fc_w, data, dstdata,FC_W_W,1,
                                  FC_W_H, 1);

    //float *tmpdata;

    //float *dev_tmpdata;
    //cudaHostGetDevicePointer((void**)&dev_tmpdata, (void*)dataOut, 0);

    ADD_G(dstdata,dev_cnn_model.CNN_fc_b,FC_W_W,dstdata);

    //cudaDeviceSynchronize();

    //cudaDeviceSynchronize

    Time = (double) cvGetTickCount() - Time;
    printf("run time11 = %gms\n", Time / (cvGetTickFrequency() * 1000));


    Time = (double) cvGetTickCount();
    //float *dd= (float *) malloc(FC_W_W * sizeof(float));
    //cudaMemcpy(dataOut,dstdata,FC_W_W * sizeof(float),cudaMemcpyDeviceToHost);

    //cudaHostGetDevicePointer(&dataOut, dstdata, 0);

//    cudaMemcpyAsync( dataOut, dstdata,
//                                 FC_W_W * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                                 stream );
//    cudaStreamSynchronize( stream ) ;


    cudaMemcpy(dataOut, dstdata, FC_W_W * sizeof(float), cudaMemcpyDefault);
//    //    cudaStreamSynchronize( stream ) ;




    Time = (double) cvGetTickCount() - Time;
    printf("run time21 = %gms\n", Time / (cvGetTickFrequency() * 1000));

    printf("dd:%d:%f\n",511,dataOut[511]);

//    for (int k = 0; k < FC_W_W; ++k) {
//        printf("dd:%d:%f\n",k,dataOut[k]);
//    }






















}


int main() {

    CNN_Model cnn_model;
    init(cnn_model);


    CNN_Model dev_cnn_model;
    dev_cnn_model.CNN_W = (float **) malloc(W_B_LEN * sizeof(float *));
    dev_cnn_model.CNN_B = (float **) malloc(W_B_LEN * sizeof(float *));
    dev_cnn_model.CNN_Prelu = (float **) malloc(W_B_LEN * sizeof(float *));

    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_w), FC_W_H * FC_W_W * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_b), FC_W_W * sizeof(float)));

    checkCudaErrors(cudaMemcpy(dev_cnn_model.CNN_fc_w, cnn_model.CNN_fc_w, sizeof(float) * FC_W_H * FC_W_W,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(dev_cnn_model.CNN_fc_b, cnn_model.CNN_fc_b, sizeof(float) * FC_W_W, cudaMemcpyHostToDevice));

    //checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.model_len), W_B_LEN * sizeof(MODEL_LEN)));
    dev_cnn_model.model_len = (MODEL_LEN *) malloc(W_B_LEN * sizeof(MODEL_LEN));
    for (int k = 0; k < W_B_LEN; ++k) {
        int k1 = cnn_model.model_len[k].k1;
        int k2 = cnn_model.model_len[k].k2;
        int in_len = cnn_model.model_len[k].in_len;
        int out_len = cnn_model.model_len[k].out_len;
        dev_cnn_model.model_len[k].k1 = k1;
        dev_cnn_model.model_len[k].k2 = k2;
        dev_cnn_model.model_len[k].in_len = in_len;
        dev_cnn_model.model_len[k].out_len = out_len;


//        checkCudaErrors(cudaMemcpy(&dev_cnn_model.model_len[k].k1, &k1, sizeof(int) * 1, cudaMemcpyHostToDevice));
//
//        checkCudaErrors(cudaMemcpy(&dev_cnn_model.model_len[k].k2, &k2, sizeof(int) * 1, cudaMemcpyHostToDevice));
//
//        checkCudaErrors(
//                cudaMemcpy(&dev_cnn_model.model_len[k].in_len, &in_len, sizeof(int) * 1, cudaMemcpyHostToDevice));
//
//        checkCudaErrors(
//                cudaMemcpy(&dev_cnn_model.model_len[k].out_len, &out_len, sizeof(int) * 1, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &(dev_cnn_model.CNN_W[k]), sizeof(float) * k1 * k2 * in_len * out_len));
        checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_B[k]), sizeof(float) * 1 * out_len));
        checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_Prelu[k]), sizeof(float) * 1 * out_len));

        checkCudaErrors(
                cudaMemcpy(dev_cnn_model.CNN_W[k], cnn_model.CNN_W[k], sizeof(float) * k1 * k2 * in_len * out_len,
                           cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dev_cnn_model.CNN_B[k], cnn_model.CNN_B[k], sizeof(float) * 1 * out_len,
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dev_cnn_model.CNN_Prelu[k], cnn_model.CNN_Prelu[k], sizeof(float) * 1 * out_len,
                                   cudaMemcpyHostToDevice));
    }


    const int WIDTH = 96;
    const int HEIGHT = 112;
    const int Channels = 3;
    const int SCALE = 512;

    CNN_Data dev_cnn_data;

    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data), sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data1), sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.dstdata), sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data_cp), sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));


    IplImage *bgr = cvLoadImage("/home/yanhao/360.bmp", 1);
    unsigned char *bgr_mat = (unsigned char *) malloc(bgr->width * bgr->height * 3);
    rgb2Mat(bgr, bgr_mat);



    int w = bgr->width;
    int h = bgr->height;
    int c = bgr->nChannels;
    float *img = (float *) malloc(sizeof(float) * w * h * c);
    float *dataIn = (float *) malloc(sizeof(float) * w * h * c);
    unsigned char *p1 = (unsigned char *) bgr_mat;
    float *p2 = img;
    for (int i = 0; i < w * h * c; ++i) {
        //float tmp = (unsigned char) (*p1++);
        *p2++ = ((unsigned char) (*p1++) - 127.5) * 0.0078125;
    }


    float *p_b = img;
    float *p_g = img + 1;
    float *p_r = img + 2;
    float *data_b = (float *) dataIn;
    float *data_g = (float *) (dataIn + w * h);
    float *data_r = (float *) (dataIn + 2 * w * h);

    for (int j = 0; j < w * h; ++j) {
        *data_b++ = *p_b;
        *data_g++ = *p_g;
        *data_r++ = *p_r;
        p_b += 3;
        p_g += 3;
        p_r += 3;
    }
    //memcpy(data, img, w * h * c * sizeof(float));



    cublasStatus_t cublasStatus;
    cublasHandle_t handle;
    cublasStatus = cublasCreate(&handle);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            //cout << "CUBLAS 对象实例化出错" << endl;
        }
        printf("CUBLAS 对象实例化出错\n");
        //getchar();
        printf("hello,is r\n");
        return -1;
    }


    const int FEA_LEN = 512;

    //float *fea;
    //cudaHostAlloc((void**)&fea,FEA_LEN * sizeof(float));
    //cudaMallocHost((void **)&fea, sizeof(float)*FEA_LEN, cudaHostAllocMapped);

    float *fea = (float *) malloc(FEA_LEN * sizeof(float));
   // double Time = (double) cvGetTickCount();
            //
            cudaStream_t stream;
            cudaStreamCreate( &stream ) ;
    runFace(handle, dataIn, w, h, c, dev_cnn_model, dev_cnn_data, fea, FEA_LEN);
    printf("%f\n",fea[511]);
    //cudaStreamSynchronize( stream ) ;


    //cudaFreeHost(fea);
   // cudaStreamDestroy( stream );

//    for (int l = 0; l < 512; ++l) {
//        printf("%f\n",fea[l]);
//    }

    //getchar();
    printf("ok!\n");
}
