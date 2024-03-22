#include <iostream>
#include <chrono>
#include "vector_imp.hpp"
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

int main(int argc, char *argv[]) {
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

    float* x = new float[n];
    float z = 0;

    for(size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i); 
    }

    double start_time = my_clock();


    for (int i = 0; i < runs; ++i) {       
        norm2(n, x, z);
    }

    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;
    std::cout << "Average execution time over " << runs << " runs: " << averageDuration << " ms" << std::endl;
    std::cout << "result: " << z << std::endl;

    delete[] x;

    return 0;
}