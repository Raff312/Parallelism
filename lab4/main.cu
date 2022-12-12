#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h> // подключение библиотеки cuBLAS
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // макрос для работы с индексами в стиле FORTRAN
int main() {
const int N = 6;
cublasHandle_t handle;
float *dev_A, *dev_b;
float *x, *A, *b;
x = (float *)malloc(N * sizeof(*x));
b = (float *)malloc(N * sizeof(*b));
A = (float *)malloc(N * N * sizeof(*A));

int ind = 11; // инициализация матрицы и вектора правой части
for (int j = 0; j < N; j++) {
for (int i = 0; i < N; i++)
if (i >= j)
A[IDX2C(i, j, N)] = (float)ind++;
else A[IDX2C(i, j, N)] = 0.0f;
b[j] = 1.0f; }
// выделяем память на GPU соответствующего размера для каждой переменной
cudaMalloc((void**)&dev_b, N * sizeof(*x));
cudaMalloc((void**)&dev_A, N * N * sizeof(*A));
cublasCreate(&handle); // инициализируем контекст cuBLAS
// копируем вектор и матрицу из CPU в GPU
cublasSetVector(N, sizeof(*b), b, 1, dev_b, 1);
cublasSetMatrix(N, N, sizeof(*A), A, N, dev_A, N);

// решаем нижнюю треугольню матрицу
cublasStrsv(handle, CUBLAS_FILL_MODE_LOWER,
CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, dev_A, N, dev_b, 1);
// копируем результат из GPU в CPU
cublasGetVector(N, sizeof(*x), dev_b, 1, x, 1);
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++)
printf("%3.0f ", A[IDX2C(i, j, N)]);
printf(" = %f %4.6f\n", b[i], x[i]); }
cudaFree(dev_b); // освобождаем память в GPU
cudaFree(dev_A);
cublasDestroy(handle); // уничтожаем контекст cuBLAS
free(x); free(b); free(A); // освобождаем память в CPU
}
