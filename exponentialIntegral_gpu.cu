#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "exponentialIntegral_gpu.cuh"

/**
 * @brief A dummy CUDA kernel placeholder (not used in final implementation).
 */
__global__ void dummy_kernel() {
    // No operation
}

/**
 * @brief Device function that evaluates the exponential integral E_n(x) in float precision.
 *
 * Implements continued fraction or series expansion depending on the value of x.
 *
 * @param n The order of the exponential integral.
 * @param x The point at which to evaluate E_n(x).
 * @param maxIterations Maximum number of iterations for convergence.
 * @return The value of E_n(x) as a float.
 */
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

/**
 * @brief Device function that evaluates the exponential integral E_n(x) in double precision.
 *
 * @param n The order of the exponential integral.
 * @param x The point at which to evaluate E_n(x).
 * @param maxIterations Maximum number of iterations for convergence.
 * @return The value of E_n(x) as a double.
 */
__device__ double exponentialIntegralDoubleDevice(const int n, const double x, const int maxIterations) {
	const double eulerConstant = 0.5772156649015329;
	const double epsilon = 1.E-30;
	const double bigdouble = 1.7976931348623157E+308; // double max
	int i, ii, nm1 = n - 1;
	double a, b, c, d, del, fact, h, psi, ans = 0.0;

	if (n == 0) return exp(-x) / x;

	if (x > 1.0) {
		b = x + n;
		c = bigdouble;
		d = 1.0 / b;
		h = d;
		for (i = 1; i <= maxIterations; i++) {
			a = -i * (nm1 + i);
			b += 2.0;
			d = 1.0 / (a * d + b);
			c = b + a / c;
			del = c * d;
			h *= del;
			if (fabs(del - 1.0) <= epsilon)
				return h * exp(-x);
		}
		return h * exp(-x);
	} else {
		ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
		fact = 1.0;
		for (i = 1; i <= maxIterations; i++) {
			fact *= -x / i;
			if (i != nm1) {
				del = -fact / (i - nm1);
			} else {
				psi = -eulerConstant;
				for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
				del = fact * (-log(x) + psi);
			}
			ans += del;
			if (fabs(del) < fabs(ans) * epsilon) return ans;
		}
		return ans;
	}
}


/**
 * @brief CUDA kernel that computes exponential integrals in parallel.
 *
 * Each thread computes one E_n(x) value and stores it into the result array.
 *
 * @param n Maximum order of E_n(x).
 * @param numberOfSamples Number of x samples in the interval [a, b].
 * @param a Left bound of interval.
 * @param b Right bound of interval.
 * @param maxIterations Maximum iterations for convergence.
 * @param result Output array of size n * numberOfSamples.
 */
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

/**
 * @brief CUDA kernel that computes exponential integrals in double precision.
 *
 * @param n Maximum order of E_n(x).
 * @param numberOfSamples Number of x samples in the interval [a, b].
 * @param a Left bound of interval.
 * @param b Right bound of interval.
 * @param maxIterations Maximum iterations for convergence.
 * @param result Output array of size n * numberOfSamples.
 */
__global__ void computeExponentialIntegralDoubleKernel(
    int n, int numberOfSamples, double a, double b, int maxIterations, double* result) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * numberOfSamples;
    if (idx >= total) return;

    int i = idx / numberOfSamples + 1;  // n starts from 1
    int j = idx % numberOfSamples + 1;  // sample index

    double x = a + ((b - a) / numberOfSamples) * j;
    result[idx] = exponentialIntegralDoubleDevice(i, x, maxIterations);
}

/*
// void launch_cuda_integral(int n, int numberOfSamples, float a, float b, int maxIterations,
//                           bool timing, bool verbose, bool useDouble,
//                           std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut, double& totalGpuTime) {
void launch_cuda_integral(unsigned  int n, unsigned  int numberOfSamples, double a, double b, int maxIterations,
                          bool timing, bool verbose, bool useDouble,
                          std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut,
                          double& totalGpuTime){

    int total = n * numberOfSamples;

    if (useDouble) {
        double* d_result;
        gpuDoubleOut.resize(total);
        cudaMalloc((void**)&d_result, sizeof(double) * total);

        int threadsPerBlock = 256;
        int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        if (timing) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }

        computeExponentialIntegralDoubleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            n, numberOfSamples, a, b, maxIterations, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(gpuDoubleOut.data(), d_result, sizeof(double) * total, cudaMemcpyDeviceToHost);

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "[CUDA] Total GPU time (double) = " << milliseconds << " ms" << std::endl;
            totalGpuTime = milliseconds / 1000.0;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        if (verbose) {
            for (int i = 0; i < total; ++i)
                std::cout << "[GPU-DOUBLE] E = " << gpuDoubleOut[i] << std::endl;
        }

        cudaFree(d_result);
    } else {
        float* d_result;
        gpuFloatOut.resize(total);
        cudaMalloc((void**)&d_result, sizeof(float) * total);

        int threadsPerBlock = 256;
        int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;
        if (timing) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        }

        // computeExponentialIntegralKernel<<<blocksPerGrid, threadsPerBlock>>>(
        //     n, numberOfSamples, a, b, maxIterations, d_result);
        computeExponentialIntegralKernel<<<blocksPerGrid, threadsPerBlock>>>(
             n, numberOfSamples, static_cast<float>(a), static_cast<float>(b), maxIterations, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(gpuFloatOut.data(), d_result, sizeof(float) * total, cudaMemcpyDeviceToHost);

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "[CUDA] Total GPU time (float) = " << milliseconds << " ms" << std::endl;
			totalGpuTime = milliseconds / 1000.0;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        if (verbose) {
            for (int i = 0; i < total; ++i)
                std::cout << "[GPU] E = " << gpuFloatOut[i] << std::endl;
        }

        cudaFree(d_result);
    }
}
*/

/**
 * @brief Launches the CUDA exponential integral computation on the GPU.
 *
 * This function dispatches either a single-precision or double-precision CUDA kernel
 * to compute E_n(x) for all combinations of n and x samples, and includes timing for
 * the entire CUDA operation (including memory allocation, kernel execution, memory copy, and free).
 *
 * @param n                Maximum order of the exponential integral.
 * @param numberOfSamples  Number of x samples in the interval [a, b].
 * @param a                Left bound of interval.
 * @param b                Right bound of interval.
 * @param maxIterations    Maximum number of iterations for convergence.
 * @param timing           If true, CUDA timing will be printed and stored.
 * @param verbose          If true, prints individual GPU results.
 * @param useDouble        If true, uses double precision. Otherwise, uses float.
 * @param gpuFloatOut      Output vector for GPU float results (used if useDouble == false).
 * @param gpuDoubleOut     Output vector for GPU double results (used if useDouble == true).
 * @param totalGpuTime     Output: total GPU time in seconds, including allocation, kernel, copy, and free.
 */
void launch_cuda_integral(unsigned int n, unsigned int numberOfSamples, double a, double b, int maxIterations,
                          bool timing, bool verbose, bool useDouble,
                          std::vector<float>& gpuFloatOut, std::vector<double>& gpuDoubleOut,
                          double& totalGpuTime) {
int total = n * numberOfSamples;

    if (useDouble) {
        double* d_result;
        gpuDoubleOut.resize(total);

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;

        if (timing) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);  // 🟡 开始计时（包住 malloc、kernel、memcpy、free）
        }

        cudaMalloc((void**)&d_result, sizeof(double) * total);

        int threadsPerBlock = 256;
        int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

        computeExponentialIntegralDoubleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            n, numberOfSamples, a, b, maxIterations, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(gpuDoubleOut.data(), d_result, sizeof(double) * total, cudaMemcpyDeviceToHost);
        cudaFree(d_result);  // ✅ 包含在计时范围内

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "[CUDA] Total GPU time (double) = " << milliseconds << " ms" << std::endl;
            totalGpuTime = milliseconds / 1000.0;  // 转为秒
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        if (verbose) {
            for (int i = 0; i < total; ++i)
                std::cout << "[GPU-DOUBLE] E = " << gpuDoubleOut[i] << std::endl;
        }

    } else {
        float* d_result;
        gpuFloatOut.resize(total);

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;

        if (timing) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);  // 🟡 开始计时
        }

        cudaMalloc((void**)&d_result, sizeof(float) * total);

        int threadsPerBlock = 256;
        int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

        computeExponentialIntegralKernel<<<blocksPerGrid, threadsPerBlock>>>(
            n, numberOfSamples, static_cast<float>(a), static_cast<float>(b), maxIterations, d_result);
        cudaDeviceSynchronize();

        cudaMemcpy(gpuFloatOut.data(), d_result, sizeof(float) * total, cudaMemcpyDeviceToHost);
        cudaFree(d_result);  // ✅ 包含在计时内

        if (timing) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "[CUDA] Total GPU time (float) = " << milliseconds << " ms" << std::endl;
            totalGpuTime = milliseconds / 1000.0;  // 转为秒
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        if (verbose) {
            for (int i = 0; i < total; ++i)
                std::cout << "[GPU] E = " << gpuFloatOut[i] << std::endl;
        }
    }
}


void test_double_kernel(int n, int numberOfSamples, double a, double b, int maxIterations) {
	int total = n * numberOfSamples;
	double* d_result;
	double* h_result = new double[total];

	cudaMalloc((void**)&d_result, sizeof(double) * total);

	int threadsPerBlock = 256;
	int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

	computeExponentialIntegralDoubleKernel<<<blocksPerGrid, threadsPerBlock>>>(
		n, numberOfSamples, a, b, maxIterations, d_result);
	cudaDeviceSynchronize();

	cudaMemcpy(h_result, d_result, sizeof(double) * total, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < numberOfSamples; ++j) {
			int idx = i * numberOfSamples + j;
			double x = a + ((b - a) / numberOfSamples) * (j + 1);
			std::cout << "[GPU-DOUBLE] E_" << (i + 1) << "(" << x << ") = "
			          << h_result[idx] << std::endl;
		}
	}

	cudaFree(d_result);
	delete[] h_result;
}
