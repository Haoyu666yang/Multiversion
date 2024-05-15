#!/bin/bash
source ./setup.sh
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed, exiting..."
    exit 1
fi


    echo "Running system-optimized blockSize..."
    # Replace ARG1 and ARG2 with your actual arguments
    ./main $1 $2 $3

