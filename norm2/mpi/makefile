# By default select gcc-13
CXX?=mpic++
CXXFLAGS?=-O3 

SOURCES=vmpi.cpp main.cpp
HEADERS=vmpi.hpp
OBJ=$(SOURCES:.cpp=.o)

all: main

vmpi.o: vmpi.cpp vmpi.hpp
	$(CXX) $(CXXFLAGS) -c vmpi.cpp -o vmpi.o

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o main

clean:
	rm -f main $(OBJ)

