#include <iostream>
#include "vector_operations.hpp"

int main() {
    const size_t n = 5;
    const float alpha = 2.0;
    float x[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[n] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float z[n];

    axpy(n, alpha, x, y, z);

    for (size_t i = 0; i < n; ++i) {
        std::cout << "z[" << i << "] = " << z[i] << std::endl;
    }

    return 0;
}