//
// Created by yanhao on 17-11-20.
//

#include "mul_blas_sse.h"
#include <immintrin.h>
#include <cblas.h>
#include <stdio.h>
//#include <xmmintrin.h>
//#include <avxintrin.h>
#include "mul_blas_sse.h"
float XR_gemm(const float *a, const float *b, int len) {

#if 0
    float sum = 0.0;
    float *p = (float *) a;
    float *q = (float *) b;
    for (int i = 0; i < len; ++i) {
        sum += (*p++) * (*q++);
    };
    return sum;
#endif
#if 0
    float s = 0;
    int nBlockWidth = 4;//SSE一次处理4个float
    int cntBlock = len / nBlockWidth;
    int cntRem = len % nBlockWidth;
    __m128 fSum = _mm_setzero_ps();//求和变量，初值清零

    float *pa = (float *) a;
    float *pb = (float *) b;



    //__m128 fa;
    //__m128 fb;
    //__m128 tmp;

    for (unsigned int i = 0; i < cntBlock; i++) {
        //fa = _mm_loadu_ps(pa);//加载
        //fb = _mm_loadu_ps(pb);


       // tmp = _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb));//求和
        fSum = _mm_add_ps( _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)), fSum);
        pa += nBlockWidth;
        pb += nBlockWidth;
    }


    const float *q = (const float *) &fSum;
    s = q[0] + q[1] + q[2] + q[3];      //合并

    for (int i = 0; i < cntRem; i++)//处理尾部剩余数据
    {
        s += pa[i+4*cntBlock] * pb[i+4*cntBlock];
    }
    return s;
#endif


#if 0
    float s = 0;
    int nBlockWidth = 8;//SSE一次处理4个float
    int cntBlock = len / nBlockWidth;
    int cntRem = len % nBlockWidth;
    //__m128 fSum = _mm_setzero_ps();//求和变量，初值清零
    float *pa = (float *) a;
    float *pb = (float *) b;
//    __m256 fa;
//    __m256 fb;
//    __m256 tmp;
    __m256 fSum = _mm256_set1_ps(0.0);
    for (unsigned int i = 0; i < cntBlock; i++) {
        //fa = _mm256_loadu_ps(pa );//加载
        //fb = _mm256_loadu_ps(pb);
        //tmp = ;//求和
        fSum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa ), _mm256_loadu_ps(pb)), fSum);
        pa += nBlockWidth;
        pb += nBlockWidth;
    }

    const float *q = (const float *) &fSum;
    s = q[0] + q[1] + q[2] + q[3] + q[4]+q[5]+q[6]+q[7];      //合并

    for (int i = 0; i < cntRem; i++)//处理尾部剩余数据
    {
        s += pa[i] * pb[i];
    }
    return s;
#endif

#if 1
    if (len < 4) {
        float sum = 0.0;
        float *p = (float *) a;
        float *q = (float *) b;
        for (int i = 0; i < len; ++i) {
            sum += (*p++) * (*q++);
        };
        return sum;
    } else if (len >= 4 && len < 8) {
        float s = 0;
        int nBlockWidth = 4;//SSE一次处理4个float
        int cntBlock = len / nBlockWidth;
        int cntRem = len % nBlockWidth;
        __m128 fSum = _mm_setzero_ps();//求和变量，初值清零
        float *pa = (float *) a;
        float *pb = (float *) b;

        for (unsigned int i = 0; i < cntBlock; i++) {
            fSum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)), fSum);
            pa += nBlockWidth;
            pb += nBlockWidth;
        }
        const float *q = (const float *) &fSum;
        s = q[0] + q[1] + q[2] + q[3];      //合并
        for (int i = 0; i < cntRem; i++)//处理尾部剩余数据
        {
            s += pa[i] * pb[i];
        }
        return s;


    } else if (len >= 8 && len < 32) {

        float s = 0;
        int nBlockWidth = 8;//SSE一次处理4个float
        int cntBlock = len / nBlockWidth;
        int cntRem = len % nBlockWidth;


        //__m128 fSum = _mm_setzero_ps();//求和变量，初值清零


        float *pa = (float *) a;
        float *pb = (float *) b;



//    __m256 fa;
//    __m256 fb;
//    __m256 tmp;
        __m256 fSum = _mm256_set1_ps(0.0);


        for (unsigned int i = 0; i < cntBlock; i++) {
            //fa = _mm256_loadu_ps(pa );//加载
            //fb = _mm256_loadu_ps(pb);


            //tmp = ;//求和
            fSum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb)), fSum);
            pa += nBlockWidth;
            pb += nBlockWidth;
        }


        const float *q = (const float *) &fSum;
        s = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];      //合并

        for (int i = 0; i < cntRem; i++)//处理尾部剩余数据
        {
            s += pa[i] * pb[i];
        }
        return s;


    } else {

        int nBlockWidth = 8 * 4;//SSE一次处理4个float
        int cntBlock = len / nBlockWidth;
        int cntRem = len % nBlockWidth;

        float *pa = (float *) a;
        float *pb = (float *) b;

        __m256 fSum = _mm256_setzero_ps();
        __m256 fSum1 = _mm256_setzero_ps();
        __m256 fSum2 = _mm256_setzero_ps();
        __m256 fSum3 = _mm256_setzero_ps();


        for (unsigned int i = 0; i < cntBlock; i++) {
            fSum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb)), fSum);
            fSum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa + 8), _mm256_loadu_ps(pb + 8)), fSum1);
            fSum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa + 16), _mm256_loadu_ps(pb + 16)), fSum2);
            fSum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pa + 24), _mm256_loadu_ps(pb + 24)), fSum3);
            pa += nBlockWidth;
            pb += nBlockWidth;
        }

        fSum = _mm256_add_ps(fSum, fSum1);
        fSum2 = _mm256_add_ps(fSum2, fSum3);
        fSum = _mm256_add_ps(fSum2, fSum);

        //fSum = _mm256_add_ps(_mm256_add_ps(fSum, fSum1), _mm256_add_ps(fSum2, fSum3));

        const float *q = (const float *) &fSum;

        float s = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];      //合并
        for (int i = 0; i < cntRem; i++)//处理尾部剩余数据
        {
            s += pa[i] * pb[i];
        }
        return s;
    }
#endif

}


//N A的列，B的列; M:A的行col K:B的行col
//C:M行K列
int run_gemm(const float *A, const float *B, float *C, int M, int K, int N, int mode) {

    if (mode == 0) {
        //const enum CBLAS_ORDER Order = CblasRowMajor;
        //const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
        //const enum CBLAS_TRANSPOSE TransB = CblasTrans;
        //const int M=4000;//A的行数，C的行数
        //const int N=300;//B的列数，C的列数
        //const int K=200;//A的列数，B的行数
        const float alpha = 1;
        const float beta = 0;
        const int lda = N;//A的列
        const int ldb = N;//B的列
        const int ldc = K;//C的列
        //const float A[M*N]={1,2,3,4,5,6,7,8,9,8,7,6};
        //const float B[K*N]={5,4,3,2,1,0};
        //float C[M*K];
        //float C1[M*K];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (mode == 1) {
        float *p_im_data = C;
        for (int l = 0; l < M; ++l) {
            int len = N;
            float *p_w = (float *) (A + l * len);
            //float b = cnnModel.pconvModel_B_O[0][l];
            //float ai = cnnModel.pRelu_O[0][l];

            float *p_data = (float *) B;
            for (int k = 0; k < K; ++k) {
                //float tmp = XR_gemm(p_w, p_data, len) + b;
                *p_im_data++ = XR_gemm(p_w, p_data, len);
                p_data += len;
            }
        }
    } else{
        printf("no this mode!\n");
    }
    return 0;
}