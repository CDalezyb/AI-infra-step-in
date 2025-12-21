#pragma once
// System includes
#include <assert.h>
#include <stdio.h>
#include <type_traits>
#include <typeinfo>
#include <string>
// CUDA runtime
#include <cuda_runtime.h>

#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

// get the elapse time of the target kernel
#define TIME_ELAPSE(func, elapsedTime)                          \
    do                                                          \
    {                                                           \
        cudaEvent_t start, stop;                                \
        CHECK(cudaEventCreate(&start));                         \
        CHECK(cudaEventCreate(&stop));                          \
        CHECK(cudaEventRecord(start, 0));                       \
        (func);                                                 \
        CHECK(cudaEventRecord(stop, 0));                        \
        CHECK(cudaEventSynchronize(stop));                      \
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop)); \
        CHECK(cudaEventDestroy(start));                         \
        CHECK(cudaEventDestroy(stop));                          \
    } while (0)

#define TIME_ELAPSE_STREAM(func, elapsedTime, stream)           \
    do                                                          \
    {                                                           \
        cudaEvent_t start, stop;                                \
        CHECK(cudaEventCreate(&start));                         \
        CHECK(cudaEventCreate(&stop));                          \
        CHECK(cudaEventRecord(start, stream));                  \
        (func);                                                 \
        CHECK(cudaEventRecord(stop, stream));                   \
        CHECK(cudaEventSynchronize(stop));                      \
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop)); \
        CHECK(cudaEventDestroy(start));                         \
        CHECK(cudaEventDestroy(stop));                          \
    } while (0)

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

template <typename T>
void initialRandomData(T *ip, int size)
{
    static bool is_seed_initialized = false;
    if (!is_seed_initialized)
    {
        srand(static_cast<unsigned>(time(nullptr)));
        is_seed_initialized = true;
    }

    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, int> || std::is_same_v<T, double>,
        "initialRandomData only supports int/float/double, Check the template parameter type.");

    if constexpr (std::is_same_v<T, int>)
    {
        for (int i = 0; i < size; ++i)
        {
            ip[i] = static_cast<int>(rand() & 0xff);
        }
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        for (int i = 0; i < size; ++i)
        {
            ip[i] = static_cast<float>(rand() & 0xffff) / 1000.0f;
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        for (int i = 0; i < size; ++i)
        {
            ip[i] = static_cast<double>(rand() & 0xff);
        }
    }
}

template <typename T>
void printMatrix(T *C, const int col_x, const int row_y)
{
    static_assert(
        std::is_same_v<T, int> || 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double>,
        "printMatrix only supports int/float/double, Check the template parameter type."
    );

    T *ic = C;
    printf("Matrix<%d, %d> (type: %s):\n", row_y, col_x,
           std::is_same_v<T, int> ? "int" :
           (std::is_same_v<T, float> ? "float" : "double"));

    // row-major
    for (int i = 0; i < row_y; i++)
    {
        for (int j = 0; j < col_x; j++)
        {
            if constexpr (std::is_same_v<T, int>)
            {
                printf("%3d ", ic[j]);
            }
            else
            {
                printf("%3.6f ", ic[j]);
            }
        }
        ic += col_x;
        printf("\n");
    }
}
