#include <iostream>
#include <chrono>
#include "vmpi.hpp"
#include <mpi.h>
#include <sys/time.h>

double my_clock()
{
/* struct timeval { long        tv_sec;
            long        tv_usec;        };
 
struct timezone { int   tz_minuteswest;
             int        tz_dsttime;      };     */
 
        struct timeval tp;
        struct timezone tzp;
        int i;
 
        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const size_t n = 20000; 

    int chunk_size = n/size;
    int start = rank*chunk_size;
    int end = (rank == size - 1) ? n : (rank+1)*chunk_size;

    ptrtdiff_t extent = end - start; 
    const float alpha = 2.0;

    float* x = new float[extent];
    float* y = new float[extent];
    float* z = new float[extent];

    for(size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i); 
        y[i] = static_cast<float>(n - i); 
    }


    const int runs = 100; //
    MPI_Barrier(comm);
    double start_time = my_clock();

    for (int i = 0; i < runs; ++i) {        
        axpy(extent, alpha, x, y, z);     
    }
    
    MPI_Barrier(comm);
    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;
    
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