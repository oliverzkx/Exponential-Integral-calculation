#ifndef EXPONENTIALINTEGRAL_GPU_CUH
#define EXPONENTIALINTEGRAL_GPU_CUH

void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations, bool timing, bool verbose);

#endif