#!/bin/bash
source ./setup.sh
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed, exiting..."
    exit 1
fi

echo "Running Emplicit version on $(uname)"
# $1 for problem size and $2 for run times
./main $1 $2
