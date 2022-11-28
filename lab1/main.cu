#include <stdio.h>
#include <random>
#include <functional>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

constexpr int N = 4096;
constexpr size_t VECTOR_SIZE = N * sizeof(double);
constexpr size_t MATRIX_SIZE = N * N * sizeof(double);
constexpr size_t BLOCK_SIZE = 256;

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<double> distr(0, 100);

void initVector(double *arr);
void initMatrix(double *arr);
void showVector(const double *arr);
void showMatrix(const double *arr);
bool checkIfVectorsEqual(const double* a, const double *b);
bool checkIfMatrcesEqual(const double* a, const double *b);
void mult(const double* a, const double*b, double* res);
void parallelMult(const double* a, const double*b, double* res);
__global__ void parallelMultKernel(const double* a, const double*b, double* res);

int main() {
    // 1

    double *a = new double[N];
    double *b = new double[N];
    double *c = new double[N * N];

    initVector(a);
    initVector(b);

    // 2

    double *ab_mult_res = new double[N];

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    mult(a, b, ab_mult_res);

    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    double elapsedTime = ms_double.count();

    double *ab_parellel_mult_res = new double[N];
    double *d_a, *d_b, *d_ab_parellel_mult_res;

    cudaMalloc((void**)&d_a, VECTOR_SIZE);
    cudaMalloc((void**)&d_b, VECTOR_SIZE);
    cudaMalloc((void**)&d_ab_parellel_mult_res, VECTOR_SIZE);

    cudaMemcpy(d_a, a, VECTOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, VECTOR_SIZE, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    float parallel_elapsed_time;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    parallelMult(d_a, d_b, d_ab_parellel_mult_res);

    cudaEventRecord(stopEvent, 0);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&parallel_elapsed_time, startEvent, stopEvent);

    cudaMemcpy(ab_parellel_mult_res, d_ab_parellel_mult_res, VECTOR_SIZE, cudaMemcpyDeviceToHost);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // write results

    std::ofstream out("output.txt");

    if (checkIfVectorsEqual(ab_mult_res, ab_parellel_mult_res)) {
        out << "Vectors are equal!" << std::endl;
    } else {
        out << "Vectors are not equal!" << std::endl;
    }

    out << std::setprecision(4) << "Time spent executing by the CPU: " << elapsedTime << " milliseconds" << std::endl;
    out << std::setprecision(4) << "Time spent executing by the GPU: " << parallel_elapsed_time << " milliseconds" << std::endl;

    out.close();

    // deallocate memory

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab_parellel_mult_res);

    delete[] a;
    delete[] b;
    delete[] c;

    delete[] ab_mult_res;
    delete[] ab_parellel_mult_res;

    return 0;
}

void initVector(double *arr) {
    for (size_t i = 0; i < N; i++) {
        arr[i] = distr(eng);
    }
}

void initMatrix(double *arr) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            arr[j * N + i] = distr(eng);
        }
    }
}

void mult(const double* a, const double*b, double* res) {
    for (size_t i = 0; i < N; i++) {
        res[i] = a[i] * b[i];
    }
}

void parallelMult(const double* a, const double*b, double* res) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parallelMultKernel<<<num_blocks, BLOCK_SIZE>>>(a, b, res);
}

__global__ void parallelMultKernel(const double* a, const double*b, double* res) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > N - 1) {
        return;
    }

    res[tid] = a[tid] * b[tid];
}

bool checkIfVectorsEqual(const double* a, const double *b) {
    for (size_t i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

bool checkIfMatrcesEqual(const double* a, const double *b) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (a[j * N + i] != b[j * N + i]) {
                return false;
            }
        }
    }

    return true;
}


void showVector(const double *arr) {
    for (size_t i = 0; i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(9) << arr[i];
    }
    std::cout << std::endl;
}

void showMatrix(const double *arr) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(9) << arr[j * N + i];
        }

        std::cout << std::endl;
    }
    std::cout << std::endl;
}