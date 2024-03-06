#include <iostream>
#include <chrono>
#include "vmpi.hpp"
#include <mpi.h>


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const size_t n = 20000; 
    const float alpha = 2.0;
    float* x = new float[n];
    float* y = new float[n];
    float* z = new float[n];

    for(size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i); 
        y[i] = static_cast<float>(n - i); 
    }

    const int runs = 100;
    double totalDuration = 0; 

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        axpy(n, 2.0, x, y, z);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        totalDuration += duration.count();
    }

    double averageDuration = totalDuration / runs;
    
    if (rank == size - 1){
    std::cout << "Average execution time over " << runs << " runs: " << averageDuration << " ms" << std::endl;
    std::cout << "result: " << z[n-1] << std::endl;
    }

    delete[] x;
    delete[] y;
    delete[] z;

    MPI_Finalize();
    return 0;
}