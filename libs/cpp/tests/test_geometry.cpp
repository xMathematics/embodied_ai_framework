#include "embodied/geometry.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    embodied::Vec3 a{0.0, 0.0, 0.0};
    embodied::Vec3 b{3.0, 4.0, 0.0};
    const double d = embodied::distance(a, b);
    assert(std::fabs(d - 5.0) < 1e-9);
    std::cout << "test_geometry: OK (distance = " << d << ")\n";
    return 0;
}
