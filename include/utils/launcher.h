#ifndef LAUNCHER_H
#define LAUNCHER_H

#include "utils/utils.h"
#include "utils/stream.h"
#include <cuda_runtime.h>

template<typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template<typename F, typename... Args>
__global__ void KernelWrapperForEach(size_t size, F f, Args... args) {
  for (size_t i = TID_1D; i < size; i += TOTAL_THREADS_1D) {
    f(i, args...);
  }
}

inline void CheckKernelLaunch(const char *func_name, const char *file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Error Launching kernel for function '%s' at %s:%d %s\n",
            func_name, file, line, cudaGetErrorString(err));
    std::terminate();
  }
}

#define CUDA_CHECK_KERNEL(func) \
  CheckKernelLaunch(#func, __FILE__, __LINE__)

template<typename F, typename... Args>
void LaunchKernel(const CudaStream &stream, dim3 grid_size, dim3 block_size, F f,
                  Args &&... args) {
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

template<typename F, typename... Args>
void LaunchKernel(const CudaStream &stream, F f, Args &&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
    &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
    reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

template<typename F, typename... Args>
void LaunchKernel(const CudaStream &stream, size_t size, F f, Args &&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

template<typename F, typename... Args>
void LaunchKernelForEach(const CudaStream &stream, size_t size, F f, Args &&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  // KernelSizingMax(grid_size, block_size, size);

  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    size, f, std::forward<Args>(args)...);

  stream.Sync();

  CUDA_CHECK_KERNEL(f);
}

template<typename F, typename... Args>
void LaunchKernelForEachMax(const CudaStream &stream, size_t size, F f, Args &&... args) {
  int grid_size, block_size;

  // KernelSizing(grid_size, block_size, size);
  KernelSizingMax(grid_size, block_size, size);

  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
    size, f, std::forward<Args>(args)...);

  stream.Sync();

  CUDA_CHECK_KERNEL(f);
}

template<typename F, typename... Args>
void LaunchKernelFix(const CudaStream &stream, size_t size, F f, Args &&... args) {
  KernelWrapper<<<256, 256, 0, stream.cuda_stream()>>>(
    f, std::forward<Args>(args)...);

  CUDA_CHECK_KERNEL(f);
}

#endif //LAUNCHER_H
