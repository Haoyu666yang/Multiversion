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

int main() {
    const size_t n = 20000; 
    const float alpha = 2.0;
    float* x = new float[n];
    float* y = new float[n];
    float* z = new float[n];

    for(size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i); 
        y[i] = static_cast<float>(n - i); 
    }

    double start_time = my_clock();
    const int runs = 100;

    for (int i = 0; i < runs; ++i) {       
        axpy(n, 2.0, x, y, z);
    }

    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;
    std::cout << "Average execution time over " << runs << " runs: " << averageDuration << " ms" << std::endl;
    std::cout << "result: " << z[n-1] << std::endl;

    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}