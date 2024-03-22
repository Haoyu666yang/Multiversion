#ifndef VMPI_HPP
#define VMPI_HPP
#include <mpi.h>
#include <cstddef>

void norm2(const size_t n,  const float* x, float& z, MPI_Comm comm, int rank);

#endif 