#include "vmpi.hpp"
#include <mpi.h>

void axpy(const size_t n, const float alpha, const float* x, const float* y, float* z) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = n/size;
    int start = rank*chunk_size;
    int end = (rank == size - 1) ? n : (rank+1)*chunk_size;

    for (size_t i = start; i < end; ++i) {
        z[i] = alpha * x[i] + y[i];
    }
}
