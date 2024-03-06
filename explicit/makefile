# By default select gcc-13
CXX=g++-13
CXXFLAGS=-mavx2 -mfma -O3

SOURCES=vector_emp.cpp main.cpp
HEADERS=vector_emp.hpp
OBJ=$(SOURCES:.cpp=.o)

all: main

vector_emp.o: vector_emp.cpp vector_emp.hpp
	$(CXX) $(CXXFLAGS) -c vector_emp.cpp -o vector_emp.o

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o main

clean:
	rm -f main $(OBJ)

