CXX?=nvcc
CXXFLAGS?=-Xcompiler -O3,-mavx2

ARCHFLAGS=\
-gencode arch=compute_50,code=sm_50 \
-gencode arch=compute_60,code=sm_60 \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_80,code=sm_80

SOURCES=$(wildcard *.cu)
HEADERS=
OBJ=$(SOURCES:.cu=.o)

all: main

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -c $< -o $@

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $^ -o $@

clean:
	rm -f main $(OBJ)

