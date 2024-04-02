#include "vmpi.hpp"
#include <mpi.h>
#include <cmath>
#ifdef __AVX2__
#include <immintrin.h>
void norm2_avx_mpi(const size_t n,  const float* x, float& z, MPI_Comm comm, int rank){

    size_t i;
    __m256 sumVec =  _mm256_setzero_ps();
    for (i = 0; i + 7 < n; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]); 
        __m256 mulVec = _mm256_mul_ps(xVec, xVec); 
        sumVec = _mm256_add_ps(sumVec, mulVec); 
    }

    float sumArray[8];
    _mm256_storeu_ps(sumArray, sumVec);
    float local_sum = 0;
    float total_sum;

    for (int j = 0; j < 8; j++)
        local_sum += sumArray[j];
    for (; i < n; ++i) 
        local_sum += x[i] * x[i];

    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, comm);
    if (rank == 0) z = std::sqrt(total_sum);

}


#endif


void norm2(const size_t n,  const float* x, float& z, MPI_Comm comm, int rank) {
#ifdef __AVX2__

    norm2_avx_mpi(n, x, z, comm, rank);

#else

    float local_sum = 0;
    float total_sum;
    for (size_t i = 0; i < n; ++i) {
        local_sum += x[i]* x[i];
    }
    MPI_Allreduce(&local_sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, comm);
    if (rank == 0) z = sqrt(total_sum);
    
#endif
}
