#ifndef ARRAY_VIEW_MULTIGPU_H
#define ARRAY_VIEW_MULTIGPU_H

#include <vector>
#include <cassert>
#include "array_view.h"

// Multi-GPU aware ArrayView using raw pointers internally for fast device access
template<typename T>
class MultiGpuArrayView {
public:
  explicit MultiGpuArrayView(T **device_ptrs, int size)
    : device_ptrs_(device_ptrs), size_(size) {
  }

  DEV_INLINE T &operator()(int peer_gpu_id, size_t idx) const {
#ifndef NDEBUG
    assert(peer_gpu_id >= 0 && peer_gpu_id < size_);
#endif
    return device_ptrs_[peer_gpu_id][idx];
  }

private:
  T **device_ptrs_;
  int size_;
};

template<typename T>
class MultiGpuArrayViewManager {
public:
  explicit MultiGpuArrayViewManager(const Vector<DVector<T> > &device_vectors)
    : num_gpus_(device_vectors.size()) {
    this->Allocate(device_vectors);
  }

  ~MultiGpuArrayViewManager() { this->Free(); }

  MultiGpuArrayViewManager(const MultiGpuArrayViewManager &) = delete;

  MultiGpuArrayViewManager &operator=(const MultiGpuArrayViewManager &) = delete;

  MultiGpuArrayViewManager(MultiGpuArrayViewManager &&) = delete;

  MultiGpuArrayViewManager &operator=(MultiGpuArrayViewManager &&) = delete;

  auto GetMultiGpuArrayView(int peer_gpu_id) -> MultiGpuArrayView<T> {
    return MultiGpuArrayView<T>(device_ptrs_vector_[peer_gpu_id], sizes_[peer_gpu_id]);
  }

private:
  void Allocate(const Vector<DVector<T> > &device_vectors) {
    Vector<T *> host_ptrs(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      host_ptrs[gpu_id] = const_cast<T *>(RawPtr(device_vectors[gpu_id]));
    }

    for (const auto &device_vector: device_vectors) {
      sizes_.emplace_back(device_vector.size());
    }

    device_ptrs_vector_.resize(num_gpus_);
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      CUDA_CHECK(cudaMalloc(&device_ptrs_vector_[gpu_id], num_gpus_ * sizeof(T*)));

      CUDA_CHECK(cudaMemcpy(device_ptrs_vector_[gpu_id],
        host_ptrs.data(),
        num_gpus_ * sizeof(T*),
        cudaMemcpyHostToDevice));
    }
  }

  void Free() {
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      CUDA_CHECK(cudaFree(device_ptrs_vector_[gpu_id]));
      device_ptrs_vector_[gpu_id] = nullptr;
    }
  }

  int num_gpus_{0};
  Vector<T **> device_ptrs_vector_;
  Vector<int> sizes_;
};

#endif //ARRAY_VIEW_MULTIGPU_H
