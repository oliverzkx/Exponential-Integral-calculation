# Compiler definitions
NVCC = nvcc        # NVIDIA CUDA compiler for .cu files
CXX  = g++         # Standard C++ compiler for .cpp files

# Compiler flags
CXXFLAGS  = -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
NVCCFLAGS = -O3

# Include paths
INCPATH = -I. -I.

# Executable name and object files
EXEC = exponentialIntegral.out
OBJS = main.o exponentialIntegral_gpu.o

# Default target: build the final executable
all: $(EXEC)

# Link all object files into the final executable
$(EXEC): $(OBJS)
	$(NVCC) -o $@ $^ -lcudart   # Use nvcc to link everything together with CUDA runtime

# Rule to compile .cpp files into .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $(INCPATH) $< -o $@

# Rule to compile .cu files into .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCPATH) $< -o $@

# Clean target: remove all object files and executable
clean:
	rm -f *.o $(EXEC)