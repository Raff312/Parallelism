#include <stdio.h>
#include <random>
#include <functional>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>

typedef double base_type;
typedef unsigned int uint;

constexpr uint N = 512;
constexpr size_t VECTOR_SIZE = N * sizeof(base_type);
constexpr size_t MATRIX_SIZE = N * N * sizeof(base_type);
constexpr uint BLOCK_SIZE = 16;

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<base_type> distr(0, 100);

void initVector(base_type* arr);
void initMatrix(base_type* arr);

bool checkIfVectorsEqual(const double* a, const double *b);

void scalarMult(const base_type* a, const base_type* b, base_type* res);
void parallelScalarMult(const base_type* a, const base_type* b, base_type* res);
__global__ void parallelScalarMultKernel(const base_type* a, const base_type* b, base_type* res);

void vectorOnMatrixMul(const base_type *a, const base_type *b, base_type *res);
void parallelVectorOnMatrixMul(const base_type* a, const base_type* b, base_type* res);
__global__ void parallelVectorOnMatrixMulKernel(const base_type* a, const base_type* b, base_type* res);

int main() {
    // 1

    base_type* a = new base_type[N];
    base_type* b = new base_type[N];
    base_type* c = new base_type[N * N];

    initVector(a);
    initVector(b);
    initMatrix(c);

    // 2

    base_type* abScalarMultRes = new base_type[N];

    auto t1 = std::chrono::high_resolution_clock::now();

    scalarMult(a, b, abScalarMultRes);

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> msDouble = t2 - t1;
    double scalarMultElapsedTime = msDouble.count();

    base_type* abParellelScalarMultRes = new double[N];
    base_type* dA, *dB, *dAbParellelScalarMultRes;

    cudaMalloc((void**)&dA, VECTOR_SIZE);
    cudaMalloc((void**)&dB, VECTOR_SIZE);
    cudaMalloc((void**)&dAbParellelScalarMultRes, VECTOR_SIZE);

    cudaMemcpy(dA, a, VECTOR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, VECTOR_SIZE, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    float parallelScalarMultElapsedTime;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    parallelScalarMult(dA, dB, dAbParellelScalarMultRes);

    cudaEventRecord(stopEvent, 0);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&parallelScalarMultElapsedTime, startEvent, stopEvent);

    cudaMemcpy(abParellelScalarMultRes, dAbParellelScalarMultRes, VECTOR_SIZE, cudaMemcpyDeviceToHost);

    // 3
    base_type* abVectorOnMatrixMultRes = new base_type[N];

    t1 = std::chrono::high_resolution_clock::now();

    vectorOnMatrixMul(a, c, abVectorOnMatrixMultRes);

    t2 = std::chrono::high_resolution_clock::now();

    msDouble = t2 - t1;
    double vectorOnMatrixElapsedTime = msDouble.count();

    base_type* abParellelVectorOnMatrixMultRes = new double[N];
    base_type* dC, *dAbParellelVectorOnMatrixMultRes;

    cudaMalloc((void**)&dC, MATRIX_SIZE);
    cudaMalloc((void**)&dAbParellelVectorOnMatrixMultRes, VECTOR_SIZE);

    cudaMemcpy(dC, c, MATRIX_SIZE, cudaMemcpyHostToDevice);

    float parallelVectorOnMatrixElapsedTime;

    cudaEventRecord(startEvent, 0);

    parallelVectorOnMatrixMul(dA, dC, dAbParellelVectorOnMatrixMultRes);

    cudaEventRecord(stopEvent, 0);

    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&parallelVectorOnMatrixElapsedTime, startEvent, stopEvent);

    cudaMemcpy(abParellelVectorOnMatrixMultRes, dAbParellelVectorOnMatrixMultRes, VECTOR_SIZE, cudaMemcpyDeviceToHost);

    // write results

    std::ofstream out("output.txt");

    if ((abScalarMultRes - abParellelScalarMultRes) < 1e-5) {
        out << "The scalar product is calculated correctly!" << std::endl;
    } else {
        out << "The scalar product is calculated incorrectly!" << std::endl;
    }

    out << std::setprecision(4) << "Scalar product: Time spent executing by the CPU: " << scalarMultElapsedTime << " milliseconds" << std::endl;
    out << std::setprecision(4) << "Scalar product: Time spent executing by the GPU: " << parallelScalarMultElapsedTime << " milliseconds" << std::endl;

    out << std::endl;

    if (checkIfVectorsEqual(abVectorOnMatrixMultRes, abParellelVectorOnMatrixMultRes)) {
        out << "The vector on matrix mult is calculated correctly!" << std::endl;
    } else {
        out << "The vector on matrix mult is calculated incorrectly!" << std::endl;
    }

    out << std::setprecision(4) << "Vector on Matrix mult: Time spent executing by the CPU: " << vectorOnMatrixElapsedTime << " milliseconds" << std::endl;
    out << std::setprecision(4) << "Vector on Matrix mult: Time spent executing by the GPU: " << parallelVectorOnMatrixElapsedTime << " milliseconds" << std::endl;

    out.close();

    // destroy

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dAbParellelScalarMultRes);

    delete[] a;
    delete[] b;
    delete[] c;

    delete[] abScalarMultRes;
    delete[] abParellelScalarMultRes;

    return 0;
}

void initVector(base_type* arr) {
    for (uint i = 0; i < N; i++) {
        arr[i] = distr(eng);
    }
}

void initMatrix(base_type* arr) {
    for (uint i = 0; i < N; i++) {
        for (uint j = 0; j < N; j++) {
            arr[j * N + i] = distr(eng);
        }
    }
}

void scalarMult(const base_type* a, const base_type* b, base_type* res) {
    base_type sum = 0;
    for (uint i = 0; i < N; i++) {
        sum += a[i] * b[i];
    }

    *res = sum;
}

void parallelScalarMult(const base_type* a, const base_type* b, base_type* res) {
    parallelScalarMultKernel<<<1, N>>>(a, b, res);
}

__global__ void parallelScalarMultKernel(const base_type* a, const base_type* b, base_type* res) {
    __shared__ base_type temp[N];
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

    __syncthreads();

    if (threadIdx.x == 0) {
        base_type sum = 0.0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }

        *res = sum;
    }
}

void vectorOnMatrixMul(const base_type *a, const base_type *b, base_type *res) {
    base_type sum = 0.0;

    for (int i = 0; i < N; i++) {
        sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += a[j] * b[j * N + i];
        }

        res[i] = sum;
    }
}

void parallelVectorOnMatrixMul(const base_type* a, const base_type* b, base_type* res) {
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(N / BLOCK_SIZE);
    parallelVectorOnMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, res);
}

__global__ void parallelVectorOnMatrixMulKernel(const base_type* a, const base_type* b, base_type* res) {
    uint j0 = blockDim.x * blockIdx.x + threadIdx.x;
    base_type sum = 0;

    for (uint k = 0; k < N; k++) {
        sum += a[k] * b[k * N + j0];
    }

    uint ind = blockDim.x * blockIdx.x + threadIdx.x;
    res[ind] = sum;
}

bool checkIfVectorsEqual(const double* a, const double *b) {
    for (int i = 0; i < N; i++) {
        if ((a[i] - b[i]) > 1e-5) {
            return false;
        }
    }

    return true;
}