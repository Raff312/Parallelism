#define N 3
#define IDX2C(i, j, ld)(((j)*(ld))+(i))
#include <math.h>
#include "cuda_runtime.h"
#include "stdio.h"
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include "curand.h"
#include "cublas_v2.h"
#pragma comment (lib, "cublas.lib")


int main() {
    float *A, *B, *D, *E, *X_1, *diag_A, *dev_D, *vect_D, *dev_E, *dev_X, *X_2, *X_vrem;
    float alpha = -1.f, al = 1.f, bet = 1.f, b, nol = 0.f;
    float eps = 0.00001;
    float eps_1 = 0.f;
    float norm_D;
    float result;
    float time = 0.0f;
    cublasHandle_t handle;

    cudaEvent_t start, stop;

    A = (float *)malloc(N * N * sizeof(*A));
    B = (float *)malloc(N*sizeof(*B));
    D = (float *)malloc(N*N*sizeof(*D));
    E = (float *)malloc(N*sizeof(*E));
    X_1 = (float *)malloc(N*sizeof(*X_1));
    diag_A = (float *)malloc(N*sizeof (*diag_A));

    printf("Matrix A:");
    printf("\n");
    for (int i=0; i<N; i++){
       for (int j=0; j<N; j++) {
           scanf("%f", &b);
           A[IDX2C(i,j, N)] = b;
           if (i == j) {
              diag_A[i] = A[IDX2C(i,j,N)];
           }
      }
    }

    printf("Vector B:");
    printf("\n");
    for (int i=0; i<N; i++){
       scanf("%f", &b);
        B[i]=b;
    }

    printf("Matrix A:");
    printf("\n");
    for (int j=0; j<N; j++){
       for (int i=0; i<N; i++){
          printf("%f ", A[IDX2C(i,j,N)]);
       }
       printf("\n");
    }

    printf("Vector B: ");
    printf("\n");
    for (int i=0; i<N; i++){
        printf("%f ", B[i]);
        printf("\n");
    }

    printf("Matrix D:");
    printf("\n");
    for (int i=0; i <N; i++ ){
        for ( int j = 0; j <N; j++ ) {
            if (j==i) {
                D[IDX2C(i,j,N)] = 0;
            } else {
                D[IDX2C(i,j,N)] = -A[IDX2C(i,j,N)] / diag_A[i];
            }

            printf("%f ", D[IDX2C(i,j,N)]);
        }

        printf("\n");
     }

    printf("diag_A: ");
    printf("\n");
    for (int i=0; i<N; i++) {
        printf("%f ", diag_A[i]);
    }
    printf("\n");

    printf("Vector E:");
    printf("\n");
    for (int i=0; i<N; i++) {
       E[i] = B[i]/diag_A[i];
       X_1[i]=E[i];
       printf("%f ", E[i]);
       printf("\n");
    }

    result=.0f;
    cudaMalloc((void**)&dev_D, N*N*sizeof(*D));
    cudaMalloc((void**)&vect_D, N*N*sizeof(*D));
    cudaMalloc((void**)&dev_E, N*sizeof(*E));
    cudaMalloc((void**)&dev_X, N*sizeof(*X_1));
    cudaMalloc((void**)&X_2, N*sizeof(*X_1));
    cudaMalloc((void**)&X_vrem, N*sizeof(*X_1));

    // инициализируем контекст cuBLAS
    cublasCreate(&handle);

    cublasSetMatrix(N, N, sizeof(*D), D, N, dev_D, N);
    cublasSetVector(N*N, sizeof(*D), D, 1, vect_D, 1);
    cublasSetVector(N, sizeof(*E), E, 1, dev_E, 1);
    cublasSetVector(N, sizeof(*X_1), X_1, 1, dev_X, 1);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cublasSnrm2(handle, N*N, vect_D, 1, &norm_D);

    printf("Norma D:");
    printf("%f", norm_D);
    printf("\n");

    if (norm_D >= 1) {
        printf("Dostatochnoe uslovie ne vypolneno");
        return 0;
    }
    else {
        eps_1=((1-norm_D)/norm_D);

        printf("%f", eps_1);
        printf("\n");

        do {
            cublasSgemv(handle, CUBLAS_OP_N, N,N, &al,  dev_D, N, dev_X, 1, &bet,  X_vrem, 1);
            cublasSaxpy(handle, N, &al,  dev_E, 1, X_vrem, 1);
            cublasScopy(handle, N, X_vrem, 1, X_2, 1);

            cublasSaxpy(handle, N, &alpha, dev_X, 1, X_vrem, 1);
            cublasSnrm2(handle, N, X_vrem, 1, &result);
            cublasScopy(handle, N, X_2, 1, dev_X, 1);
            cublasSscal(handle, N, &nol, X_vrem, 1);
        }
        while (result>eps);

        cublasGetVector(N, sizeof(*dev_X), dev_X, 1, X_1, 1 );
        cublasDestroy(handle);

        cudaFree(dev_D);
        cudaFree(dev_E);
        cudaFree(dev_X);
        cudaFree(X_2);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        printf("X=");
        printf("\n");
        for (int i=0; i<N; i++) {
            printf ("%f", X_1[i]);
            printf ("\n");
        }
        printf ("GPU time: %.0f\n", time);

        return 0;
    }
}
