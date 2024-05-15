#!/bin/bash
source ./setup.sh
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed, exiting..."
    exit 1
fi

# for i in 64 256 512 1024; do
    echo "Running with optimizied blockSize..."
    # Replace ARG1 and ARG2 with your actual arguments
    ./main $1 $2 $3
    # ./main $1 $2 $i
# done
