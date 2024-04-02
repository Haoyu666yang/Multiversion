#!/bin/bash

OS="$(uname)"
MAC_COM="g++-13"
LINUX_COM="g++"

setup_linux(){
    echo "Detected Linux, checking for OpenMP compiler..."
    if command -v $LINUX_COM >/dev/null 2>&1; then
        echo "OpenMP compiler found: $($LINUX_COM --version | head -n 1)"
        export CXX=$LINUX_COM
        export CXXFLAGS='-O3 -fopenmp -mavx2'
    else
        echo "OpenMP compiler not found, please install OpenMP."
        exit 1
    fi
}

setup_mac(){
    echo "Detected macOS, checking for OpenMP compiler..."
    if command -v $MAC_COM >/dev/null 2>&1; then
        echo "OpenMP compiler found: $($MAC_COM --version | head -n 1)"
        export CXX=$MAC_COM
        export CXXFLAGS='-O3 -fopenmp'
    else
        echo "OpenMP compiler not found, please install OpenMP."
        exit 1
    fi
}

case "$OS" in
    "Linux") setup_linux;;
    "Darwin") setup_mac;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "Setup completed."

