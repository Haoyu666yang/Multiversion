#include "vector_emp.hpp"
#include <cmath>
#include <stdio.h>
#ifdef __AVX2__
#include <immintrin.h> 

void norm2_avx(const size_t n, const float* x, float& z) {

    size_t i;
    __m256 sumVec =  _mm256_setzero_ps();
    for (i = 0; i + 7 < n; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]); 
        __m256 mulVec = _mm256_mul_ps(xVec, xVec); 
        sumVec = _mm256_add_ps(sumVec, mulVec); 
    }

    float sumArray[8];
    _mm256_storeu_ps(sumArray, sumVec);
    float sum = 0;
    for (int j = 0; j < 8; j++)
        sum += sumArray[j];
    // for (i = 0; i < 8; i++)
    //     sum += sumArray[i];

    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    z = std::sqrt(sum);
}
#endif

void norm2(const size_t n, const float* x, float& z) {
#ifdef __AVX2__
    norm2_avx(n, x, z);
#else
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    z = std::sqrt(sum);
#endif
}