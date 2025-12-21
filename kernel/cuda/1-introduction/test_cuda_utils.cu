#include <iostream>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"

__global__ void testKernel(int *arr, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        arr[idx] = idx;
}

int test_TIME_ELAPSE(int size)  // 改为传入任意 size
{
    int *d_arr = nullptr;
    CHECK(cudaMalloc(&d_arr, size * sizeof(int)));
    int *h_arr = new int[size];

    // ========== 核心修正：动态计算 grid/block 维度 ==========
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // 1. default stream elapsed time
    float elapsedTime;
    TIME_ELAPSE((testKernel<<<grid_size, block_size>>>(d_arr, size)), elapsedTime);
    printf("Size = %d, Default stream time: %.3f ms\n", size, elapsedTime);
    CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\t验证：h_arr[0] = %d, h_arr[%d] = %d\n", h_arr[0], size-1, h_arr[size-1]);

    // 2. custom stream elapsed time
    cudaStream_t customStream;
    CHECK(cudaStreamCreate(&customStream));
    float streamTime;
    TIME_ELAPSE_STREAM((testKernel<<<grid_size, block_size, 0, customStream>>>(d_arr, size)), streamTime, customStream);
    printf("Size = %d, Custom stream time: %.3f ms\n", size, streamTime);
    CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\t验证：h_arr[0] = %d, h_arr[%d] = %d\n", h_arr[0], size-1, h_arr[size-1]);
    
    delete[] h_arr;

    CHECK(cudaStreamDestroy(customStream));
    CHECK(cudaFree(d_arr));
    return 0;
}

int main()
{
    // 测试任意 size：小尺寸、大尺寸、非 256 倍数的尺寸
    test_TIME_ELAPSE(100);        // 小尺寸（<256）
    test_TIME_ELAPSE(1 << 20);    // 大尺寸（1048576）
    test_TIME_ELAPSE(123456);     // 非 256 倍数的尺寸
    return 0;
}