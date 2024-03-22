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

    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << " <problem_size> <run_times>\n";
        return 1;
    }

    const size_t n = std::atoi(argv[1]);
    const int runs = std::atoi(argv[2]);
    
    if (runs <= 0 || n <= 0){
        std::cerr << "Please enter positive integers for the problem size and run times.\n";
        return 1;
    }        

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
 

    int chunk_size = n/size;
    int start = rank*chunk_size;
    int end = (rank == size - 1) ? n : (rank+1)*chunk_size;

    ptrdiff_t extent = end - start; 

    float* x = new float[extent];
    float z = 0;

    for(size_t i = start; i < end; ++i) {
        x[i-start] = static_cast<float>(i); 
    }


    // MPI_Barrier(comm);
    double start_time = my_clock();

    for (int i = 0; i < runs; ++i) {        
        norm2(extent, x, z, comm, rank);  
    }
    
    // MPI_Barrier(comm);
    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;
    

    if (rank == 0){
    std::cout << "Average execution time over " << runs << " runs: " << averageDuration << " ms" << std::endl;
    std::cout << "result: " << z << std::endl;
    }

    delete[] x;

    MPI_Finalize();
    return 0;
}