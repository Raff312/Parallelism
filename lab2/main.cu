#include <stdio.h>
#include <random>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>

#define BASE_TYPE double

constexpr size_t BLOCK_SIZE = 16;

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<BASE_TYPE> distr(0, 100);

void initMatrix(BASE_TYPE *arr, size_t oldRows, size_t oldCols, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (i > (oldRows - 1) || j > (oldCols - 1)) {
                arr[i * cols + j] = 0;
            } else {
                arr[i * cols + j] = distr(eng);
            }
        }
    }
}

int toMultiple(int a, int b) {
    int mod = a % b;
    if (mod != 0) {
        mod = b - mod;
        return a + mod;
    }

    return a;
}

__global__ void parallelMatrixMulKernel(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *c, size_t aCols, size_t bCols) {
    size_t i0 = aCols * (blockDim.y * blockIdx.y + threadIdx.y);
    size_t j0 = blockDim.x * blockIdx.x + threadIdx.x;
    BASE_TYPE sum = 0;

    for (size_t k = 0; k < aCols; k++) {
        sum += a[i0 + k] * b[k * bCols + j0];
    }

    size_t ind = bCols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    c[ind] = sum;
}

void parallelMatrixMul(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *c, size_t aRows, size_t aCols, size_t bCols) {
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(bCols / BLOCK_SIZE, aRows / BLOCK_SIZE);
    parallelMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, aCols, bCols);
}

__global__ void parallelMatrixMulSharedKernel(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *c, size_t aCols, size_t bCols) {
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    size_t aBegin = aCols * BLOCK_SIZE * by;
    size_t aEnd = aBegin + aCols - 1;

    size_t aStep = BLOCK_SIZE;
    size_t bBegin = BLOCK_SIZE * bx;

    size_t bStep = BLOCK_SIZE * bCols;
    BASE_TYPE sum = 0;
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];

        as[ty][tx] = a[ia + aCols * ty + tx];
        bs[ty][tx] = b[ib + bCols * ty + tx];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += as[ty][k] * bs[k][tx];
        }

        __syncthreads();
    }

    size_t ic = bCols * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic + bCols * ty + tx] = sum;
}

void parallelMatrixMulShared(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *c, size_t aRows, size_t aCols, size_t bCols) {
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(bCols / BLOCK_SIZE, aRows / BLOCK_SIZE);
    parallelMatrixMulSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, aCols, bCols);
}

int main() {
    cudaEvent_t startEvent1, stopEvent1;
    cudaEventCreate(&startEvent1);
    cudaEventCreate(&stopEvent1);

    cudaEvent_t startEvent2, stopEvent2;
    cudaEventCreate(&startEvent2);
    cudaEventCreate(&stopEvent2);

    size_t oldARows = 1024;
    size_t oldACols = 1024;
    size_t oldBRows = oldACols;
    size_t oldBCols = 1024;

    size_t aRows = toMultiple(oldARows, BLOCK_SIZE);
    size_t bRows = toMultiple(oldBRows, BLOCK_SIZE);
    size_t aCols = toMultiple(oldACols, BLOCK_SIZE);
    size_t bCols = toMultiple(oldBCols, BLOCK_SIZE);

    size_t aSize = aRows * aCols * sizeof(BASE_TYPE);
    size_t bSize = bRows * bCols * sizeof(BASE_TYPE);
    size_t cSize = aRows * bCols * sizeof(BASE_TYPE);

    BASE_TYPE *hA = new BASE_TYPE[aRows * aCols];
    BASE_TYPE *hB = new BASE_TYPE[bRows * bCols];
    BASE_TYPE *hC = new BASE_TYPE[aRows * bCols];
    BASE_TYPE *hCShared = new BASE_TYPE[aRows * bCols];

    initMatrix(hA, oldARows, oldACols, aRows, aCols);
    initMatrix(hB, oldBRows, oldBCols, bRows, bCols);

    BASE_TYPE *dA = nullptr;
    cudaMalloc((void**)&dA, aSize);

    BASE_TYPE *dB = nullptr;
    cudaMalloc((void**)&dB, bSize);

    BASE_TYPE *dC = nullptr;
    cudaMalloc((void**)&dC, cSize);

    BASE_TYPE *dCShared = nullptr;
    cudaMalloc((void**)&dCShared, cSize);

    cudaMemcpy(dA, hA, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, aSize, cudaMemcpyHostToDevice);

    cudaEventRecord(startEvent1, 0);

    parallelMatrixMul(dA, dB, dC, aRows, aCols, bCols);

    cudaEventRecord(stopEvent1, 0);
    cudaEventSynchronize(stopEvent1);

    cudaMemcpy(hC, dC, cSize, cudaMemcpyDeviceToHost);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, startEvent1, stopEvent1);

    cudaEventRecord(startEvent2, 0);

    parallelMatrixMulShared(dA, dB, dCShared, aRows, aCols, bCols);

    cudaEventRecord(stopEvent2, 0);
    cudaEventSynchronize(stopEvent2);

    cudaMemcpy(hC, dCShared, cSize, cudaMemcpyDeviceToHost);

    float kernelTimeShared;
    cudaEventElapsedTime(&kernelTimeShared, startEvent2, stopEvent2);

    // write results

    bool isEqual = true;
    for (size_t i = 0; i < aRows; i++) {
        for (size_t j = 0; j < bCols; j++) {
            BASE_TYPE sum = 0;
            for (size_t k = 0; k < aCols; k++) {
                sum += hA[i * aCols + k] * hB[k * bCols + j];
            }

            if (fabs(sum - hC[i * bCols + j]) > 1e-5) {
                isEqual = false;
                break;
            }
        }
    }

    if (isEqual) {
        std::cout << "Martices are equal!" << std::endl;
    } else {
        std::cout << "Matrices are not equal!" << std::endl;
    }

    std::cout << std::setprecision(4) << "Time spent executing by the GPU: " << kernelTime << " milliseconds" << std::endl;
    std::cout << std::setprecision(4) << "Time spent executing by the GPU (shared): " << kernelTimeShared << " milliseconds" << std::endl;

    // deallocate memory

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    cudaEventDestroy(startEvent1);
    cudaEventDestroy(stopEvent1);
    cudaEventDestroy(startEvent2);
    cudaEventDestroy(stopEvent2);

    return 0;
}
