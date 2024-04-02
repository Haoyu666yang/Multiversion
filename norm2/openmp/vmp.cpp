#include "vmp.hpp"
#include <omp.h>
#include <cmath>
#ifdef __AVX2__
#include <immintrin.h>
void norm2_avx_omp(const size_t n, const float* x, float& z){

    float globalSum = 0;
#pragma omp parallel reduction(+:globalSum)
{

    __m256 sumVec = _mm256_setzero_ps();

#pragma omp for nowait
    for (size_t i = 0; i<n-7; i += 8){
        __m256 loadVec = _mm256_loadu_ps(&x[i]);
        __m256 mulVec = _mm256_mul_ps(loadVec, loadVec);
        sumVec = _mm256_add_ps(mulVec, sumVec);
    }

    float sumArray[8];
    _mm256_storeu_ps(sumArray, sumVec);
    for(int j = 0; j < 8; j++)
        globalSum += sumArray[j];
}

    for (size_t i = n - (n%8); i<n; i++)
        globalSum += x[i]*x[i];

    z = std::sqrt(globalSum);

}


#endif

void norm2(const size_t n, const float* x, float& z) {

#ifdef __AVX2__

    norm2_avx_omp(n, x, z);

#else
    
    float sum = 0.0;
#pragma omp parallel  
    {
#pragma omp for reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += x[i] * x[i];
    }
    }
    z = std::sqrt(sum);

#endif
}
