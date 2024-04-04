#!/bin/bash

OS="$(uname)"
MAC_COM="g++"
LINUX_COM="g++"

setup_linux(){
    echo "Detected Linux, checking for AVX support..."
    if command -v $LINUX_COM >/dev/null 2>&1; then
        echo "Compiler found: $($LINUX_COM --version | head -n 1)"
        export CXX=$LINUX_COM
        export CXXFLAGS='-O3 -mavx2'
    else
        echo "Compiler not found, please ensure g++ is installed."
        exit 1
    fi
}

setup_mac(){
    echo "Detected macOS. Setting up without AVX flags..."
    if command -v $MAC_COM >/dev/null 2>&1; then
        echo "Compiler found: $($MAC_COM --version | head -n 1)"
        export CXX=$MAC_COM
        export CXXFLAGS='-O3'
    else
        echo "Compiler not found, please ensure g++ is installed."
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
