#include "cuda_utils.hpp"

__global__ void printThreadIdxKernel(int *mat, const int col_x, const int row_y)
{
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int idx = tid_x + tid_y * col_x;
    printf("thread_id: (%d,%d), block_id: (%d,%d), coordinate: (%d,%d), "
           "global index: %2d, mat ival: %2d\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y,
           tid_x, tid_y, idx, mat[idx]);
}

int main()
{
    initDevice(0);
    const int rowNum = 6, colNum = 8;
    const int nElements = rowNum * colNum;
    const int nBytes = nElements * sizeof(int);

    int *mat_host = (int *)malloc(nBytes);
    for (int i = 0; i < nElements; i++)
        mat_host[i] = i;

    printMatrix(mat_host, colNum, rowNum);

    int *mat_dev = nullptr;
    CHECK(cudaMalloc((void **)&mat_dev, nBytes));
    CHECK(cudaMemcpy(mat_dev, mat_host, nBytes, cudaMemcpyHostToDevice));

    dim3 blockSize(4, 2);
    dim3 girdSize((colNum - 1) / blockSize.x + 1, (rowNum - 1) / blockSize.y + 1);
    printThreadIdxKernel<<<girdSize, blockSize>>>(mat_dev, colNum, rowNum);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(mat_dev));
    free(mat_host);
    cudaDeviceReset();
    return 0;
}