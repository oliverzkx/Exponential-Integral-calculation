#pragma once
#include <vector>        
#include <iostream> 

#ifndef EXPONENTIALINTEGRAL_GPU_CUH
#define EXPONENTIALINTEGRAL_GPU_CUH

//void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations, bool timing, bool verbose);
//void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations, bool timing, bool verbose, bool useDouble);
// void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations,
//                           bool timing, bool verbose, bool useDouble,
//                           std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut);

// void launch_cuda_integral(
//     unsigned int n, 
//     unsigned int numberOfSamples, 
//     double a, 
//     double b, 
//     int maxIterations, 
//     bool timing, 
//     bool verbose, 
//     bool useDouble,
//     std::vector<float>& gpuFloatOut,
//     std::vector<double>& gpuDoubleOut,
//     double& totalGpuTime
// );

// void launch_cuda_integral(int n, int numberOfSamples, double a, double b, int maxIterations,
//                           bool timing, bool verbose, bool useDouble,
//                           std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut, double& totalGpuTime);

void launch_cuda_integral(unsigned int n, unsigned int numberOfSamples, double a, double b, int maxIterations,
                          bool timing, bool verbose, bool useDouble,
                          std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut, double& totalGpuTime);

void test_double_kernel(int n, int numberOfSamples, double a, double b, int maxIterations);

#endif