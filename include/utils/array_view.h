#ifndef ARRAY_VIEW_H
#define ARRAY_VIEW_H

#include "utils/types.h"
#include "utils/utils.h"

template<typename T>
class ArrayView {
public:
  ArrayView() = default;

  ArrayView(const DVector<T> &vec)
    : data_(const_cast<T *>(RawPtr(vec))),
      size_(vec.size()) {
  }

  ArrayView(const HVector<T> &vec)
    : data_(const_cast<T *>(RawPtr(vec))),
      size_(vec.size()) {
  }

  ArrayView(const Vector<T> &vec)
    : data_(const_cast<T *>(RawPtr(vec))),
      size_(vec.size()) {
  }

  DEV_HOST
  ArrayView(T *data, size_t size) : data_(data), size_(size) {
  }

  DEV_HOST_INLINE T *data() { return data_; }

  DEV_HOST_INLINE const T *data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_HOST_INLINE T &operator[](size_t i) {
    assert(i < size_);
    return data_[i]; // Correctly returns modifiable reference
  }

  DEV_HOST_INLINE T &operator[](size_t i) const {
    assert(i < size_);
    return data_[i]; // Returns non-modifiable reference (const)
  }

  DEV_HOST_INLINE void Swap(ArrayView<T> &rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE T *begin() { return data_; }

  DEV_HOST_INLINE T *end() { return data_ + size_; }

  DEV_HOST_INLINE const T *begin() const { return data_; }

  DEV_HOST_INLINE const T *end() const { return data_ + size_; }

private:
  T *data_{};
  size_t size_{};
};

template<typename T>
class ConstArrayView {
public:
  ConstArrayView() = default;

  ConstArrayView(const DVector<T> &vec)
    : data_(const_cast<T *>(RawPtr(vec))),
      size_(vec.size()) {
  }

  ConstArrayView(const HVector<T> &vec)
    : data_(const_cast<T *>(RawPtr(vec))),
      size_(vec.size()) {
  }

  DEV_HOST
  ConstArrayView(T *data, size_t size) : data_(data), size_(size) {
  }

  DEV_HOST_INLINE T *data() { return data_; }

  DEV_HOST_INLINE const T *data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_HOST_INLINE T &operator[](size_t i) {
    assert(i < size_);
    return data_[i]; // Correctly returns modifiable reference
  }

  DEV_HOST_INLINE const T &operator[](size_t i) const {
    assert(i < size_);
    return data_[i]; // Returns non-modifiable reference (const)
  }

  DEV_HOST_INLINE void Swap(ConstArrayView<T> &rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE T *begin() { return data_; }

  DEV_HOST_INLINE T *end() { return data_ + size_; }

  DEV_HOST_INLINE const T *begin() const { return data_; }

  DEV_HOST_INLINE const T *end() const { return data_ + size_; }

private:
  T *data_{};
  size_t size_{};
};
#endif //ARRAY_VIEW_H
