#include <iostream>
#include <cuda_runtime.h>
#include "exponentialIntegral_gpu.cuh"

__global__ void dummy_kernel() {
    
}

__device__ float exponentialIntegralFloatDevice(const int n, const float x, const int maxIterations) {
	const float eulerConstant = 0.5772156649015329f;
	const float epsilon = 1.E-30f;
	const float bigfloat = 3.402823466e+38F; // float max
	int i, ii, nm1 = n - 1;
	float a, b, c, d, del, fact, h, psi, ans = 0.0f;

	if (n == 0) return expf(-x) / x;

	if (x > 1.0f) {
		b = x + n;
		c = bigfloat;
		d = 1.0f / b;
		h = d;
		for (i = 1; i <= maxIterations; i++) {
			a = -i * (nm1 + i);
			b += 2.0f;
			d = 1.0f / (a * d + b);
			c = b + a / c;
			del = c * d;
			h *= del;
			if (fabsf(del - 1.0f) <= epsilon)
				return h * expf(-x);
		}
		return h * expf(-x);
	} else {
		ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
		fact = 1.0f;
		for (i = 1; i <= maxIterations; i++) {
			fact *= -x / i;
			if (i != nm1) {
				del = -fact / (i - nm1);
			} else {
				psi = -eulerConstant;
				for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
				del = fact * (-logf(x) + psi);
			}
			ans += del;
			if (fabsf(del) < fabsf(ans) * epsilon) return ans;
		}
		return ans;
	}
}

__global__ void computeExponentialIntegralKernel(
    int n, int numberOfSamples, float a, float b, int maxIterations, float* result) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * numberOfSamples;
    if (idx >= total) return;

    int i = idx / numberOfSamples + 1;  // n starts from 1
    int j = idx % numberOfSamples + 1;  // sample index

    float x = a + ((b - a) / numberOfSamples) * j;
    result[idx] = exponentialIntegralFloatDevice(i, x, maxIterations);
}

void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations, bool timing, bool verbose) {
    int total = n * numberOfSamples;
    float* d_result;
    float* h_result = new float[total];

    cudaMalloc((void**)&d_result, sizeof(float) * total);

    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    // CUDA timing events
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    // Launch the kernel
    computeExponentialIntegralKernel<<<blocksPerGrid, threadsPerBlock>>>(n, numberOfSamples, a, b, maxIterations, d_result);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_result, d_result, sizeof(float) * total, cudaMemcpyDeviceToHost);

    if (timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "[CUDA] Total GPU time (including kernel + memcpy): " << milliseconds << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    if (verbose) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < numberOfSamples; ++j) {
                int idx = i * numberOfSamples + j;
                float x = a + ((b - a) / numberOfSamples) * (j + 1);
                std::cout << "[GPU] E_" << (i + 1) << "(" << x << ") = "
                          << h_result[idx] << std::endl;
            }
        }
    }

    cudaFree(d_result);
    delete[] h_result;
}
