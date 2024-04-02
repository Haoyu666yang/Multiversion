#include "vector_imp.hpp"
#include <cmath>

void norm2(const size_t n, const float* x, float& z) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    z = std::sqrt(sum);

}
