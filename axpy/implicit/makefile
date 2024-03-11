# By default select gcc-13
CXX=g++-13
CXXFLAGS=-O3 

SOURCES=vector_imp.cpp main.cpp
HEADERS=vector_imp.hpp
OBJ=$(SOURCES:.cpp=.o)

all: main

vector_imp.o: vector_imp.cpp vector_imp.hpp
	$(CXX) $(CXXFLAGS) -c vector_imp.cpp -o vector_imp.o

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o main

clean:
	rm -f main $(OBJ)

