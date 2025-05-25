#include <iostream>
#include <cuda_runtime.h>
#include "exponentialIntegral_gpu.cuh"

__global__ void dummy_kernel() {
    
}

void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations, bool timing, bool verbose) {
    if (verbose) {
        std::cout << "[CUDA] Launching dummy kernel with n=" << n << ", samples=" << numberOfSamples << std::endl;
    }

    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    if (verbose) {
        std::cout << "[CUDA] Dummy kernel finished." << std::endl;
    }
}
