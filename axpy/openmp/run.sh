#!/bin/bash
source ./setup.sh

make

if [ $? -ne 0 ]; then
    echo "Compilation failed, exiting..."
    exit 1
fi

for i in $(seq 1 12); do
    echo "Running with $i thread(s)..."
    # $1 for problem size and $2 for run times
    OMP_NUM_THREADS=$i OMP_PROC_BIND=true ./main $1 $2
done
