# Exponential Integral CUDA Acceleration

This project computes the exponential integral \( E_n(x) \) using both **CPU (double precision)** and **GPU (CUDA)** with support for **float** and **double precision**. The GPU implementation achieves significant speedups and demonstrates performance tuning with several advanced CUDA techniques.

---

## 📁 Project Structure

- `main.cpp` — Main driver: parses args, runs CPU/GPU computations
- `exponentialIntegral_gpu.cu` — CUDA kernel + device logic + stream/const memory, etc.
- `exponentialIntegral_gpu.cuh` — Header file declaring CUDA interface
- `Makefile` — Build script using nvcc + g++
- `README.md` — This file

---

## 🚀 Features

- ✅ CPU baseline (double precision)
- ✅ GPU float & double precision versions
- ✅ Timing includes full CUDA pipeline: malloc + kernel + memcpy + free
- ✅ Automatic speedup reporting (CPU time / GPU time)
- ✅ Error checking: compares GPU result with CPU reference
- ✅ Supports command-line arguments for custom input sizes
- ✅ Supports both `n == m` and `n ≠ m` cases

---

## ⚙️ Build Instructions

Make sure you have an NVIDIA GPU and CUDA installed. Then run:

`make`

This will compile main.cpp and exponentialIntegral_gpu.cu into an executable named:

`./exponentialIntegral.out`

## 🧪 Usage

```bash
./exponentialIntegral.out -n 5000 -m 5000 -t
```

### 📌 Command-line Options

| Option       | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `-a <value>` | Set the **left bound** `a` of the interval a,ba, ba,b (default: `0.0`) |
| `-b <value>` | Set the **right bound** `b` of the interval a,ba, ba,b (default: `10.0`) |
| `-n <size>`  | Set the number of **orders** `n` to compute (default: `10`)  |
| `-m <size>`  | Set the number of **samples** within the interval (default: `10`) |
| `-i <value>` | Set the number of **iterations** in the integral computation (default: `2000000000`) |
| `-c`         | **Skip CPU computation**, run GPU only                       |
| `-g`         | **Skip GPU computation**, run CPU only                       |
| `-d`         | Use **double precision** on GPU (default is float)           |
| `-t`         | **Print timing info** for each execution                     |
| `-v`         | **Verbose mode**, print all individual results               |
| `-h`         | Show **help message**                                        |



### 📊 Example Runs

```bash
# Run CPU only
./exponentialIntegral.out -n 5000 -m 5000 -t -g

# Run only GPU (float)
./exponentialIntegral.out -n 5000 -m 5000 -t -c

# Run GPU with double precision
./exponentialIntegral.out -n 5000 -m 5000 -t -c -d

# Run both CPU and GPU (float), compare and print speedup
./exponentialIntegral.out -n 5000 -m 5000 -t

# Run both CPU and GPU (double), compare and print speedup
./exponentialIntegral.out -n 5000 -m 5000 -t -d
```





## 📈 Speedup Analysis

- GPU (float) achieves ~**250x** speedup vs CPU for large inputs.
- GPU (double) achieves ~**50x** speedup.
- Performance scales well with `n`, `m`.



## 🧠 Optimization Techniques Used

| Optimization           | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| ✅ Shared Memory        | Used to reduce redundant computation of `x` per thread       |
| ✅ Constant Memory      | Used to store scalar parameters (`a`, `b`, `maxIterations`) on device |
| ✅ CUDA Streams         | Split memory-copy & kernel into chunks with async overlap    |
| 🛠️ Threads/Block Tuning | Tuned performance for 32–1024 threads per block to find optimal value |



All optimization results are measured with timing (via `cudaEvent_t`) including:

- Memory allocation
- Kernel launch
- Data transfer back to host
- Memory deallocation



## 🧪 Tested Input Sizes

- `-n 5000 -m 5000`
- `-n 8192 -m 8192`
- `-n 16384 -m 16384`
- `-n 20000 -m 20000`
- ✅ Also tested `n ≠ m` configurations such as `-n 8000 -m 4000`, `-n 5000 -m 3000`



## ✅ Output Example (float GPU vs CPU)

```
[CUDA] Total GPU time (float) = 18.29 ms
calculating the exponentials on the cpu took: 4.62 seconds
Speedup (CPU / GPU) = 253.05x
```



## 📄 Notes

This README serves as a usage guide and technical summary for the CUDA implementation.  
**This is not the final report** — please refer to **`report.pdf`** for the complete project report, including detailed analysis, discussion, and conclusions.