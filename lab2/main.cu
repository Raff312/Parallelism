#include <stdio.h>
#include <random>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>

typedef double base_type;
typedef unsigned int uint;

constexpr uint BLOCK_SIZE = 16;

std::random_device rd;
std::default_random_engine eng(rd());
std::uniform_real_distribution<base_type> distr(0, 100);

void initMatrix(base_type *arr, uint oldRows, uint oldCols, uint rows, uint cols);

uint toMultiple(uint a, uint b);

void parallelMatrixMul(const base_type *a, const base_type *b, base_type *res, uint aRows, uint aCols, uint bCols);
__global__ void parallelMatrixMulKernel(const base_type *a, const base_type *b, base_type *res, uint aCols, uint bCols);

void parallelMatrixMulShared(const base_type *a, const base_type *b, base_type *res, uint aRows, uint aCols, uint bCols);
__global__ void parallelMatrixMulSharedKernel(const base_type *a, const base_type *b, base_type *res, uint aCols, uint bCols);

int main() {
    uint oldARows = 1024;
    uint oldACols = 423;
    uint oldBRows = oldACols;
    uint oldBCols = 666;

    uint aRows = toMultiple(oldARows, BLOCK_SIZE);
    uint aCols = toMultiple(oldACols, BLOCK_SIZE);
    uint bRows = toMultiple(oldBRows, BLOCK_SIZE);
    uint bCols = toMultiple(oldBCols, BLOCK_SIZE);

    base_type *hA = new base_type[aRows * aCols];
    base_type *hB = new base_type[bRows * bCols];
    base_type *hC = new base_type[aRows * bCols];

    initMatrix(hA, oldARows, oldACols, aRows, aCols);
    initMatrix(hB, oldBRows, oldBCols, bRows, bCols);

    size_t aSize = aRows * aCols * sizeof(base_type);
    size_t bSize = bRows * bCols * sizeof(base_type);
    size_t cSize = aRows * bCols * sizeof(base_type);

    base_type *dA = nullptr;
    cudaMalloc((void**)&dA, aSize);

    base_type *dB = nullptr;
    cudaMalloc((void**)&dB, bSize);

    base_type *dC = nullptr;
    cudaMalloc((void**)&dC, cSize);

    cudaMemcpy(dA, hA, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bSize, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    parallelMatrixMul(dA, dB, dC, aRows, aCols, bCols);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(hC, dC, cSize, cudaMemcpyDeviceToHost);

    base_type *hCShared = new base_type[aRows * bCols];

    base_type *dCShared = nullptr;
    cudaMalloc((void**)&dCShared, cSize);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);

    cudaEventRecord(startEvent, 0);

    parallelMatrixMulShared(dA, dB, dCShared, aRows, aCols, bCols);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(hCShared, dCShared, cSize, cudaMemcpyDeviceToHost);

    float kernelTimeShared;
    cudaEventElapsedTime(&kernelTimeShared, startEvent, stopEvent);

    // write results

    bool isEqual = true;
    for (uint i = 0; i < aRows; i++) {
        for (uint j = 0; j < bCols; j++) {
            base_type sum = 0.0;
            for (uint k = 0; k < aCols; k++) {
                sum += hA[i * aCols + k] * hB[k * bCols + j];
            }

            if (fabs(sum - hC[i * bCols + j]) > 1e-5 || fabs(sum - hCShared[i * bCols + j]) > 1e-5) {
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

    // destroy

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dCShared);

    delete[] hA;
    delete[] hB;
    delete[] hC;
    delete[] hCShared;

    return 0;
}

void initMatrix(base_type *arr, uint oldRows, uint oldCols, uint rows, uint cols) {
    for (uint i = 0; i < rows; i++) {
        for (uint j = 0; j < cols; j++) {
            if (i > (oldRows - 1) || j > (oldCols - 1)) {
                arr[i * cols + j] = 0;
            } else {
                arr[i * cols + j] = distr(eng);
            }
        }
    }
}

uint toMultiple(uint a, uint b) {
    uint mod = a % b;
    if (mod != 0) {
        mod = b - mod;
        return a + mod;
    }

    return a;
}

void parallelMatrixMul(const base_type *a, const base_type *b, base_type *res, uint aRows, uint aCols, uint bCols) {
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(bCols / BLOCK_SIZE, aRows / BLOCK_SIZE);
    parallelMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, res, aCols, bCols);
}

__global__ void parallelMatrixMulKernel(const base_type *a, const base_type *b, base_type *res, uint aCols, uint bCols) {
    uint yi = blockDim.y * blockIdx.y + threadIdx.y;

    uint ai = aCols * yi;
    uint bi = blockDim.x * blockIdx.x + threadIdx.x;

    base_type sum = 0.0;
    for (uint k = 0; k < aCols; k++) {
        sum += a[ai + k] * b[k * bCols + bi];
    }

    uint ind = bCols * yi + bi;
    res[ind] = sum;
}

void parallelMatrixMulShared(const base_type *a, const base_type *b, base_type *res, uint aRows, uint aCols, uint bCols) {
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(bCols / BLOCK_SIZE, aRows / BLOCK_SIZE);
    parallelMatrixMulSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, res, aCols, bCols);
}

__global__ void parallelMatrixMulSharedKernel(const base_type *a, const base_type *b, base_type *res, uint aCols, uint bCols) {
    uint aBegin = aCols * blockDim.y * blockIdx.y;
    uint aEnd = aBegin + aCols - 1;
    uint aStep = blockDim.x;

    uint bBegin = blockDim.x * blockIdx.x;
    uint bStep = blockDim.y * bCols;

    __shared__ base_type as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ base_type bs[BLOCK_SIZE][BLOCK_SIZE];

    base_type sum = 0.0;
    for (uint ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep) {
        as[threadIdx.y][threadIdx.x] = a[ia + aCols * threadIdx.y + threadIdx.x];
        bs[threadIdx.y][threadIdx.x] = b[ib + bCols * threadIdx.y + threadIdx.x];

        __syncthreads();

        for (uint k = 0; k < blockDim.x; k++) {
            sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    uint ind = bCols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    res[ind] = sum;
}
