#pragma once

namespace embodied::cuda {

// 在 GPU 上执行 C = A + B (每个线程处理一个元素)
// 调用方需保证 a/b/c 均为设备端指针且长度 >= n
void vectorAdd(const float* a, const float* b, float* c, int n);

}  // namespace embodied::cuda
