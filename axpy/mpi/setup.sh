#!/bin/bash

OS="$(uname)"
MAC_COM="mpic++"
LINUX_COM="mpic++"

setup_linux(){
    echo "Detected Linux, checking for MPI compiler..."
    if command -v $LINUX_COM >/dev/null 2>&1; then
        echo "MPI compiler found: $($LINUX_COM --version | head -n 1)"
        export CXX=$LINUX_COM
        export CXXFLAGS='-O3'
    else
        echo "MPI compiler not found, please install MPI."
        exit 1
    fi
}

setup_mac(){
    echo "Detected macOS, checking for MPI compiler..."
    if command -v $MAC_COM >/dev/null 2>&1; then
        echo "MPI compiler found: $($MAC_COM --version | head -n 1)"
        export CXX=$MAC_COM
        export CXXFLAGS='-O3'
    else
        echo "MPI compiler not found, please install MPI."
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

