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

#define W_B_LEN 20
#define W_B_Data_Dim 4
#define FC_W_H 21504
#define FC_W_W 512


#define SAFE_FREE(p)      {if((p) != NULL) {free(p); (p) = NULL;}}
#define SAFE_CLOSE(fp)    {if((fp) != NULL) {fclose((fp)); (fp) = NULL;}}

//float *img = (float *) malloc(sizeof(float) * WIDTH * HEIGHT * Channels);
//float *data = (float *) malloc(sizeof(float) * WIDTH * HEIGHT * Channels * 512);   //将图片的像素值，复制进网络的输入Blob
//float *data1 = (float *) malloc(512 * WIDTH * HEIGHT * Channels * sizeof(float));
//float *dstdata = (float *) malloc(512 * WIDTH * HEIGHT * sizeof(float));
//float *data_cp = (float *) malloc(512 * WIDTH * HEIGHT * Channels * sizeof(float));



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

//    for (int l = 0; l < cnn_model.model_len[19].out_len; ++l) {
//        printf("%f\n",cnn_model.CNN_Prelu[19][l]);
//    }




    SAFE_CLOSE(fp_cnn_len);
    SAFE_CLOSE(fp_cnn_w);
    SAFE_CLOSE(fp_cnn_b);
    SAFE_CLOSE(fp_cnn_prelu);
    SAFE_CLOSE(fp_cnn_fc_w);
    SAFE_CLOSE(fp_cnn_fc_b);
    printf("init ok!\n");
    return 0;
}





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
        if ((i + 1) % W == 0 && i != 0) printf("\n");
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


cublasStatus_t
addWithCuda3(const cublasHandle_t &handle, float *c, const float *a, const float *b, unsigned int WA, unsigned int HA,
             unsigned int WB,
             unsigned int HB) {

    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;


    printf("aaaaaaaaaaa!\n");
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        // Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_a, HA * WA * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void **) &dev_b, HB * WB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }

    //cublasSetVector(HA * WA, sizeof(float), a,1, dev_a, 1);
    //cublasSetVector(HB * WB, sizeof(float), b, 1, dev_b, 1);

    cudaStatus = cudaMemcpy(dev_a, &a, HA * WA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMemcpy(dev_b, &b, HB * WB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }
    cudaStatus = cudaMemset(dev_c, 0, sizeof(HB * HA));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }

    // 同步函数
    //cudaThreadSynchronize();

    float alpha = 1.0;
    float beta = 0.0;
    clock_t start = clock();

    printf("%d,%d,%d,%d\n", HA, HB, WA, WB);
    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HB, HA, WB, &alpha, dev_b, HA, dev_a, HA, &beta, dev_c,
                               HA);

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
//        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
//            printf("CUBLAS 对象实例化出错\n");
//        }

        printf("errror!\n");
    }


    clock_t time_used = clock() - start;
    printf("(GPU31) time:%ld\n", time_used);
    //cudaThreadSynchronize();
    //cublasGetVector(HA * WB, sizeof(float), c, 1, dev_c, 1);

    cudaStatus = cudaMemcpy(c, dev_c, HA * HB * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!");
        //goto Error;
    }

    //Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    printf("ok!\n");
    return cublasStatus;
}

void im2col(const float *data_im, const int channels, int height, int width, const int kszie,
            const int pad, const int stride,
            float *data_col, int *h_out, int *w_out) {

    int height_col = (height + 2 * pad -
                      ((kszie - 1) + 1)) / stride + 1;
    int width_col = (width + 2 * pad -
                     ((kszie - 1) + 1)) / stride + 1;

    *h_out = height_col;
    *w_out = width_col;
    float *pcol_1, *pcol_2;
    float *pimg_1;

    int c, h, w, k1, k2;
    for (c = 0; c < channels; ++c) {
        pcol_1 = data_col + c * kszie * kszie;
        pimg_1 = (float *) (data_im + width * height * c);
        int a = 0;
        for (h = -pad; h < height_col * stride - pad; h += stride) {
            for (w = -pad; w < width_col * stride - pad; w += stride) {
                pcol_2 = pcol_1 + a * channels * kszie * kszie;
                a++;
                for (k1 = 0; k1 < kszie; ++k1) {
                    for (k2 = 0; k2 < kszie; ++k2) {
                        if (h + k1 < 0 || w + k2 < 0 || h + k1 > height - 1 || w + k2 > width - 1) {
                            *pcol_2++ = 0;
                        } else {
                            *pcol_2++ = *(pimg_1 + (h + k1) * width + (w + k2));
                        }
                    }
                }
            }
        }
    }
}


inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu(const float *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                float *data_col) {
    const int output_h = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
                          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}


void checkCudaErrors(cudaError_t code) {
    if (code != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(code) << std::endl;
        exit(-1);
//        if( abort )
//            exit( code );
    }
}

int main() {

#if 0
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
    cudaStatus = cudaSetDevice(0);
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

//    typedef struct _CNN_Model {
//        MODEL_LEN *model_len;
//        float **CNN_W;
//        float **CNN_B;
//        float **CNN_Prelu;
//        float *CNN_fc_w;
//        float *CNN_fc_b;
//    } CNN_Model;

//    int len[W_B_LEN * W_B_Data_Dim];
//    MODEL_LEN model_len[W_B_LEN];
//    fread(len, sizeof(int), W_B_LEN * W_B_Data_Dim, fp_cnn_len);
//
//    for (int i = 0; i < W_B_LEN; ++i) {
//        model_len[i].k1 = len[W_B_Data_Dim * i];
//        model_len[i].k2 = len[W_B_Data_Dim * i + 1];
//        model_len[i].in_len = len[W_B_Data_Dim * i + 2];
//        model_len[i].out_len = len[W_B_Data_Dim * i + 3];
//    }
////
//    cnn_model.model_len[l].k1 = model_len[l].k1;
//    cnn_model.model_len[l].k2 = model_len[l].k2;
//    cnn_model.model_len[l].in_len = model_len[l].in_len;
//    cnn_model.model_len[l].out_len = model_len[l].out_len;

#if 0

    CNN_Model cnn_model;
    init(cnn_model);

    cudaError_t cudaStatus;
    cublasStatus_t cublasStatus;


    CNN_Model dev_cnn_model;
    //dev_cnn_model.model_len = (MODEL_LEN *) malloc(W_B_LEN * sizeof(MODEL_LEN));
    dev_cnn_model.CNN_W = (float **) malloc(W_B_LEN * sizeof(float *));
    dev_cnn_model.CNN_B = (float **) malloc(W_B_LEN * sizeof(float *));
    dev_cnn_model.CNN_Prelu = (float **) malloc(W_B_LEN * sizeof(float *));


    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_w), FC_W_H * FC_W_W * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_b), FC_W_W * sizeof(float)));

    //dev_cnn_model.CNN_fc_w= (float *) (FC_W_H * FC_W_W * sizeof(float));
    //dev_cnn_model.CNN_fc_b= (float *) (FC_W_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(dev_cnn_model.CNN_fc_w, cnn_model.CNN_fc_w, sizeof(float) * FC_W_H * FC_W_W,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(dev_cnn_model.CNN_fc_b, cnn_model.CNN_fc_b, sizeof(float) * FC_W_W, cudaMemcpyHostToDevice));
//
//

    //checkCudaErrors(cudaMalloc((void **) &dev_cnn_model, 1 * sizeof(CNN_Model)));
    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.model_len), W_B_LEN * sizeof(MODEL_LEN)));
//    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_W), W_B_LEN * sizeof(float *)));
//    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_B), W_B_LEN * sizeof(float *)));
//    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_Prelu), W_B_LEN * sizeof(float *)));
//    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_w), FC_W_H * FC_W_W * sizeof(float)));
//    checkCudaErrors(cudaMalloc((void **) (&dev_cnn_model.CNN_fc_b), FC_W_W * sizeof(float)));
//
//    checkCudaErrors(cudaMemcpy(dev_cnn_model.CNN_fc_w, cnn_model.CNN_fc_w, sizeof(float) * FC_W_H * FC_W_W,
//                               cudaMemcpyHostToDevice));
//    checkCudaErrors(
//            cudaMemcpy(dev_cnn_model.CNN_fc_b, cnn_model.CNN_fc_b, sizeof(float) * FC_W_W, cudaMemcpyHostToDevice));


    for (int k = 0; k < W_B_LEN; ++k) {
        int k1 = cnn_model.model_len[k].k1;
        int k2 = cnn_model.model_len[k].k2;
        int in_len = cnn_model.model_len[k].in_len;
        int out_len = cnn_model.model_len[k].out_len;

        checkCudaErrors(cudaMemcpy(&dev_cnn_model.model_len[k].k1, &k1, sizeof(int) * 1, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(&dev_cnn_model.model_len[k].k2, &k2, sizeof(int) * 1, cudaMemcpyHostToDevice));

        checkCudaErrors(
                cudaMemcpy(&dev_cnn_model.model_len[k].in_len, &in_len, sizeof(int) * 1, cudaMemcpyHostToDevice));

        checkCudaErrors(
                cudaMemcpy(&dev_cnn_model.model_len[k].out_len, &out_len, sizeof(int) * 1, cudaMemcpyHostToDevice));

        //       printf("aaa\n");
        //float *tt =dev_cnn_model.CNN_W[k];
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

        //fread(cnn_model.CNN_W[k], sizeof(float), k1 * k2 * in_len * out_len, fp_cnn_w);
        //fread(cnn_model.CNN_B[k], sizeof(float), 1 * out_len, fp_cnn_b);
        //fread(cnn_model.CNN_Prelu[k], sizeof(float), 1 * out_len, fp_cnn_prelu);
    }


    const int WIDTH = 96;
    const int HEIGHT = 112;
    const int Channels = 3;
    const int SCALE = 512;

    CNN_Data dev_cnn_data;

    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data),     sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data1),    sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.dstdata),  sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));
    checkCudaErrors(cudaMalloc((void **) &(dev_cnn_data.data_cp),  sizeof(float) * SCALE * WIDTH * HEIGHT * Channels));





    //float *img = (float *) malloc(sizeof(float) * WIDTH * HEIGHT * Channels);
//    float *data = (float *) malloc(sizeof(float) * WIDTH * HEIGHT * Channels * 512);   //将图片的像素值，复制进网络的输入Blob
//    float *data1 = (float *) malloc(512 * WIDTH * HEIGHT * Channels * sizeof(float));
//    float *dstdata = (float *) malloc(512 * WIDTH * HEIGHT * sizeof(float));
//    float *data_cp = (float *) malloc(512 * WIDTH * HEIGHT * Channels * sizeof(float));
//


    getchar();

    printf("ok!\n");

//    cudaMalloc((void **) &(dev_cnn_model->model_len), 1 * sizeof(MODEL_LEN));
//    cudaMalloc((void **) &(dev_cnn_model->CNN_W), 1 * sizeof(float **));
//    cudaMalloc((void **) &(dev_cnn_model->CNN_B), 1 * sizeof(float **));
//    cudaMalloc((void **) &(dev_cnn_model->CNN_Prelu), 1 * sizeof(float **));
//    cudaMalloc((void **) &(dev_cnn_model->CNN_fc_w), 1 * sizeof(float *));
//    cudaMalloc((void **) &(dev_cnn_model->CNN_fc_b), 1 * sizeof(float *));
//
//    cudaMalloc((void **) &(dev_cnn_model->model_len), 1 * sizeof(MODEL_LEN));





    // Allocate GPU buffers for three vectors (two input, one output)    .
    //cudaStatus = cudaMalloc((void **) &dev_c, HA * WB * sizeof(float));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        return CUBLAS_STATUS_INTERNAL_ERROR;
//
//    }


#endif

#if 1

    const int WA = 3;
    const int HA = 5;

    const int WB = 4;
    const int HB = 3;

    float A[WA * HA] = {1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                        2,0,2,
                        3,5,1
    };

    float B[WB * HB] = {1, 3, 9,1,
                        2, 1, 1,2,
                        0, 2, 8,0
    };

    const int WC=WB;
    const int HC=HA;

    float C[100] = {0};


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

    cublasStatus = addWithCuda6(handle, A, B, WA, HA, WB, HB, C);

    printMatrix(A, WA, HA);
    printf("\n");
    printMatrix(B, WB, HB);
    printf("\n");
    printMatrix(C, WC, HC);

    cublasDestroy(handle);


#endif

#if 0

    const int WA = 30;
    const int HA = 10;
    const int WB = 30;
    const int HB = 50;
    printf("aaa\n");

//    float A[WA * HA] = {1, 2, 3,
//                        4, 5, 6};
//
//    float B[WB * HB] = {1, 3, 9,
//                        2, 1, 1,
//                        3, 2, 5,
//                        0, 2, 8
//    };


    float AA[25] = {100, 10, 12, 10, 1,
                    2, 3, 5, 6, 7,
                    12, 2, 21, 22, 11,
                    100, 21, 20, 200, 11,
                    109, 102, 194, 11, 21};

    int h_out, w_out;


    float AB[1000];
    //im2col2(AA, 1, 5, 5, 2, 0, 1, AB, &h_out, &w_out);
    printf("%d,%d\n", h_out, w_out);

    im2col_cpu(AA,3,5,5,2,2,0,0,1,1,1,1,AB);

    for (int y = 0; y < 2 * 2; ++y) {
        for (int x = 0; x < 4 * 4; ++x) {
            printf("%f ", AB[y * 16 + x]);
        }
        printf("\n");
    }


    float *B = (float *) calloc(HB * WB, sizeof(float));
    float *A = (float *) calloc(HA * WA, sizeof(float));
    float *C = (float *) calloc(HA * HB, sizeof(float));
    if (!B || !A || !C) {
        printf("err!\n");
        exit(-1);
    }
    //float *D = (float *) calloc(HA * WB, sizeof(float));
    //float *E = (float *) calloc(HA * WB, sizeof(float));
    //float *F = (float *) calloc(height_A * width_B, sizeof(float));


    // memset(A, 0.0, sizeof(float) * height_A * width_A);
    //memset(B, 0.0, sizeof(float) * height_B * width_B);)
    printf("aaa\n");
    //float *C = (float *) malloc(HA * HB * sizeof(float));

    memset(C, 0.0, HA * HB * sizeof(float));
    srand((unsigned) clock());
    randomInit(B, HB * WB);
    randomInit(A, HA * WA);

    cudaError_t cudaStatus;
    //cublasStatus_t cublasStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }
    cublasStatus_t cublasStatus;

    cublasHandle_t handle;
    cublasStatus = cublasCreate(&handle);


    printf("ccccccccccccccccc!\n");
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        if (cublasStatus == CUBLAS_STATUS_NOT_INITIALIZED) {
            printf("CUBLAS 对象实例化出错\n");
        }
        printf("CUBLAS 对象实例化出错\n");

        return -1;
    }
    printf("ccccccccccccccccc!\n");

    cublasStatus = addWithCuda5(handle, C, A, B, WA, HA, WB, HB);

//    printMatrix(A, WA, HA);
//    printMatrix(B, WB, HB);
//    printMatrix(C, HB, HA);



    cublasDestroy(handle);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


#endif
    return 0;
}
