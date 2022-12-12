#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

#include <cuda_runtime.h>
#include <cufft.h>

typedef float2 Complex;
#define N_ELEM 10
int
main(int argc, char **argv)
{
    Complex *h_signal;
    Complex *d_signal;
    int mem_size;
    cufftHandle plan;

    printf("[truely simpleCUFFT] is starting...\n");
    //Выделяем память для входных данных на хосте
    h_signal = (Complex *)malloc(sizeof(Complex) * N_ELEM);

    //Инициализируем входные данные
    for (unsigned int i = 0; i < N_ELEM; ++i){
        h_signal[i].x = (float)i;
        h_signal[i].y = 0;
        printf("%f\t%f\n", h_signal[i].x, h_signal[i].y);
    }

    //Выделяем память для входных данных на видеокарте и копируем их туда
    mem_size = sizeof(Complex) * N_ELEM;
    cudaMalloc((void **)&d_signal, mem_size);
    cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);

    //Собственно выполняем преобразование Фурье
    printf("Transforming signal cufftExecC2C\n");
    cufftPlan1d(&plan, N_ELEM, CUFFT_C2C, 1);
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

    //Копируем результат вычисления с видеокарты на хост
    cudaMemcpy(h_signal, d_signal, mem_size, cudaMemcpyDeviceToHost);
    for (unsigned int i = 0; i < N_ELEM; ++i)
        printf("%f\t%f\n", h_signal[i].x, h_signal[i].y);

    //Прибираем за собой
    cufftDestroy(plan);
    free(h_signal);
    cudaFree(d_signal);
    cudaDeviceReset();

    exit(0);
}