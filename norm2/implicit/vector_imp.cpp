#include "vector_imp.hpp"

void norm2(const size_t n, const float* x, float& z) {
    for (size_t i = 0; i < n; ++i) {
        z += x[i] * x[i];
    }
}
