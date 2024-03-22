#include "vmpi.hpp"
#include <mpi.h>
#include <cmath>

void norm2(const size_t n,  const float* x, float& z, MPI_Comm comm, int rank) {
    float local_sum = 0;
    float total_sum;
    for (size_t i = 0; i < n; ++i) {
        local_sum += x[i]* x[i];
    }
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    if (rank == 0)
        z = sqrt(total_sum);
}
