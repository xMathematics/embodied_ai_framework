#pragma once

#include <cmath>

namespace embodied {

// 三维向量
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double norm() const { return std::sqrt(x * x + y * y + z * z); }
};

// 计算两点欧氏距离
double distance(const Vec3& a, const Vec3& b);

}  // namespace embodied
