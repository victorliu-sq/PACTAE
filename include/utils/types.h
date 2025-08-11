#ifndef TYPES_H
#define TYPES_H

#include <queue>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#include <filesystem>
#include <fstream>
#include <sstream>


// Standard vector (host-side standard memory)
template<typename T>
using Vector = std::vector<T>;

// Host-side pinned vector (accelerates H2D transfer)
using MrPinned = thrust::system::cuda::universal_host_pinned_memory_resource;

template<typename T>
using HVector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, MrPinned> >;

// template<typename T>
// using HVector = thrust::host_vector<T>; // Normal host vector, NOT pinned memory

// Device-side CUDA vector
template<typename T>
using DVector = thrust::device_vector<T>;

using PreferenceLists = Vector<Vector<int> >;

using FlatPreferenceList = Vector<int>;      // length = n * n

template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename T>
using SPtr = std::shared_ptr<T>;

template<typename T>
using Queue = std::queue<T>;

using Matching = std::vector<int>;

#define Move(x) std::move(x)

template<typename T, typename... Args>
static inline auto MakeUPtr(Args &&... args) -> UPtr<T> {
  return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename... Args>
static inline auto MakeSPtr(Args &&... args) -> SPtr<T> {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

struct PRNode {
  int idx_;
  int rank_;
};

using index_t = uint64_t;
using ns_t = uint64_t;
using ms_t = double;

using PRMatrix = std::vector<std::vector<PRNode> >;

using String = std::string;

// ========================== Stream Aliases ==============================
using OutStringStream = std::ostringstream;
using InStringStream = std::istringstream;

using OutFStream = std::ofstream;
using InFStream = std::ifstream;

// ========================== Filesystem Operations ==============================
#define FileExists(x) std::filesystem::exists(x)
#define MakeDirec(x) std::filesystem::create_directories(x)
#define RemoveFile(x) std::filesystem::remove(x)           // Removes a single file
#define RemoveAll(x) std::filesystem::remove_all(x)        // Removes files and directories recursively

// =========================== SMP Workload ======================================
enum WorkloadType {
  PERFECT,
  RANDOM,
  CONGESTED,
  SOLO
};

inline String WorkloadTypeToString(WorkloadType type) {
  switch (type) {
    case PERFECT: return "Perfect";
    case RANDOM: return "Random";
    case SOLO: return "Solo";
    case CONGESTED: return "Congested";
    default: return "Unknown";
  }
}

#endif //TYPES_H
