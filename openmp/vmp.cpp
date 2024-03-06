#include "vmp.hpp"
#include <omp.h>

void axpy(const size_t n, const float alpha, const float* x, const float* y, float* z) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        z[i] = alpha * x[i] + y[i];
    }
}
