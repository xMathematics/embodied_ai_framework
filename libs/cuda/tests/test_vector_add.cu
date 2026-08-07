#include "embodied/cuda/vector_add.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 1024;
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f * i;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    embodied::cuda::vectorAdd(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        assert(h_c[i] == h_a[i] + h_b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::printf("test_vector_add: OK (%d elements)\n", n);
    return 0;
}
