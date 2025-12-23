mkdir -p build
# test cuda_utils.hpp
nvcc -std=c++17 ./test_cuda_utils.cu -o ./build/test_cuda_utils -lcudart
./build/test_cuda_utils

# print threadIdx and global idx of each thread
nvcc -std=c++17 ./print_thread_index.cu -o ./build/print_thread_index -lcudart
./build/print_thread_index