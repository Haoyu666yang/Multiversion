#include "vmp.hpp"
#include <omp.h>
#include <cmath>

void norm2(const size_t n, const float* x, float& z) {
    float sum = 0.0;
#pragma omp parallel  
    {
#pragma omp for reduction(+:z)
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    }
    z = std::sqrt(sum);
}
