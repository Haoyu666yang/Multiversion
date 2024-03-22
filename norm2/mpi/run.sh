#!/bin/bash
source ./setup.sh

make

if [ $? -ne 0 ]; then
    echo "Compilation failed, exiting..."
    exit 1
fi

for i in $(seq 1 12); do
    echo "Running with $i MPI process(es)..."
    # Replace ARG1 and ARG2 with your actual arguments
    mpirun -np $i ./main $1 $2
done
