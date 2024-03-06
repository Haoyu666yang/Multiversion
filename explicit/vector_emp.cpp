#include "vector_emp.hpp"
#include <immintrin.h> 

void axpy(const size_t n, const float alpha, const float* x, const float* y, float* z) {
    size_t i;
    __m256 alphaVec = _mm256_set1_ps(alpha); 
    for (i = 0; i + 7 < n; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]); 
        __m256 yVec = _mm256_loadu_ps(&y[i]); 
        __m256 resultVec = _mm256_fmadd_ps(xVec, alphaVec, yVec); 
        _mm256_storeu_ps(&z[i], resultVec); 
    }
    for (; i < n; ++i) {
        z[i] = alpha * x[i] + y[i];
    }
}
