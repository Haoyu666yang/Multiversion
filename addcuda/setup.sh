#!/bin/bash

OS="$(uname)"
MAC_COM="g++"
LINUX_COM="nvcc"

setup_linux(){
    echo "Detected Linux, checking for AVX support..."
    if command -v $LINUX_COM >/dev/null 2>&1; then
        echo "Compiler found: $($LINUX_COM --version | grep release)"
        export CXX=$LINUX_COM
        export CXXFLAGS='-Xcompiler -O3,-mavx2'
    else
        echo "Compiler not found, please ensure nvcc is installed."
        exit 1
    fi
}

setup_mac(){
    echo "Detected macOS. No CUDA supported"
    exit 1
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
