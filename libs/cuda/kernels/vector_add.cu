#include "embodied/cuda/vector_add.hpp"

#include <cuda_runtime.h>

namespace embodied::cuda {

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vectorAdd(const float* a, const float* b, float* c, int n) {
    constexpr int kBlockSize = 256;
    const int numBlocks = (n + kBlockSize - 1) / kBlockSize;
    vectorAddKernel<<<numBlocks, kBlockSize>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

}  // namespace embodied::cuda
