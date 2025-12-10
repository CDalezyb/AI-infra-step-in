/**
 *  PyTorch extension cuda example: sum array.
 *  Author: kevin.xie
 *  Email: kaiyuanxie@yeah.net
 * */
#include <cmath>
#include "sumArray.h"

// THREAD_PER_BLOCK: threads per block

// gpu global memory，所有block 共享
__device__ int countSHM = 0;
__global__ void arraySumWithSHMKernel(float *arrData, const int dataSize)
{
    // shared memory，shared by threads in the same block
    // each block 有自己的 shm，block之间的shm物理独立、完全隔离
    __shared__ float shm[THREAD_PER_BLOCK];
    int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // initialize countSHM by the first
    if (thIdx == 0) {
        countSHM = 0;
        // 确保 “清零 countSHM” 的写操作，对其他所有线程都生效
        __threadfence();
    }
    // val: register variable -> local memory
    float val = 0.0;
    while (thIdx < dataSize) {
        val += arrData[thIdx];
        thIdx += blockDim.x * gridDim.x;
    }
    shm[threadIdx.x] = val;
    __syncthreads();

    // block level reduction in shared memory
    for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
        if (threadIdx.x < i)
            shm[threadIdx.x] += shm[threadIdx.x + i];
        __syncthreads();
    }

    __syncthreads();
    bool isLast = false;
    thIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIdx.x == 0) {
        arrData[blockIdx.x] = shm[0];
        __threadfence();
        int value = atomicAdd(&countSHM, 1);
        isLast = (value == gridDim.x - 1);
    }
    isLast = __syncthreads_or(isLast);
    if (isLast) {
        shm[threadIdx.x] = threadIdx.x < gridDim.x ? arrData[threadIdx.x] : 0;
        __syncthreads();
        for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
            if (threadIdx.x < i)
                shm[threadIdx.x] += shm[threadIdx.x + i];
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0)
            arrData[0] = shm[0];
    }
    __syncthreads();
}

void arraySumCUDA(float *arrData, const int dataSize) {
    int grid = max(dataSize / THREAD_PER_BLOCK, 1);
    arraySumWithSHMKernel<<<grid, THREAD_PER_BLOCK>>>(arrData, dataSize);
}

