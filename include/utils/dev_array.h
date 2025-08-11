#ifndef ARRAY_VIEW_H
#define ARRAY_VIEW_H

#include "dev_array.h"
#include "utils/utils.h"

template<typename T>
using HArray = HVector<T>;

template<typename T>
class DArrayView {
public:
  DEV_HOST_INLINE DArrayView(T *data, size_t size)
    : data_(data), size_(size) {
  }

  DEV_INLINE T &operator[](size_t idx) {
    assert(idx < size_);
    return data_[idx];
  }

  DEV_INLINE const T &operator[](size_t idx) const {
    assert(idx < size_);
    return data_[idx];
  }

  DEV_INLINE T *data() const { return data_; }

  DEV_INLINE size_t size() const { return size_; }

private:
  T *data_;
  size_t size_;
};

template<typename T>
class DArray : public DVector<T> {
public:
  // Default constructor
  DArray() : DVector<T>() {
  }

  // Constructor with size
  explicit DArray(size_t size) : DVector<T>(size) {
  }

  // Constructor with size and default value
  DArray(size_t size, const T &val) : DVector<T>(size, val) {
  }

  DArrayView<T> DeviceView() {
    return DArrayView<T>(thrust::raw_pointer_cast(this->data()), this->size());
  }

  void Fill(T val) {
    thrust::fill(this->begin(), this->end(), val);
  }

  void GetFromDeviceDirectly(HArray<T> &host_array) const {
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(this->begin(), this->end(), host_array.begin());
  }

  void GetFromDevice(HArray<T> &host_array) const {
    // Ensure host array size matches device array size clearly
    host_array.resize(this->size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(this->begin(), this->end(), host_array.begin());
  }

  void GetFromDevice(Vector<T> &host_array) const {
    // Ensure host array size matches device array size clearly
    host_array.resize(this->size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(this->begin(), this->end(), host_array.begin());
  }

  void SetToDevice(const HArray<T> &host_array) {
    // Ensure host array size matches device array size clearly
    this->resize(host_array.size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(host_array.begin(), host_array.end(), this->begin());
  }

  void SetToDevice(const Vector<T> &host_array) {
    // Ensure host array size matches device array size clearly
    this->resize(host_array.size());
    // Explicitly copy data from device (DArray) to host (HArray)
    thrust::copy(host_array.begin(), host_array.end(), this->begin());
  }
};

#endif //ARRAY_VIEW_H
