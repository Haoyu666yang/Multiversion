# By default select gcc-13
CXX=g++-13
CXXFLAGS=-O3 -fopenmp

SOURCES=vmp.cpp main.cpp
HEADERS=vmp.hpp
OBJ=$(SOURCES:.cpp=.o)

all: main

vmp.o: vmp.cpp vmp.hpp
	$(CXX) $(CXXFLAGS) -c vmp.cpp -o vmp.o

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o main

clean:
	rm -f main $(OBJ)
