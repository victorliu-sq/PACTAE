#ifndef GENERATE_WORKLOADS_FLAT_H
#define GENERATE_WORKLOADS_FLAT_H

#include <glog/logging.h>

#include <algorithm>
#include <random>
#include "stopwatch.h"
#include "types.h"
#include <filesystem>
#include <thread>

// ======================= CONGESTED ==========================
static FlatPreferenceList GeneratePrefListsCongestedFlat(int n) {
  FlatPreferenceList flat(n * n);

  Vector<int> row(n);
  for (int i = 0; i < n; ++i) row[i] = i;

  // Shuffle the single row pattern once
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(row.begin(), row.end(), gen);

  const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
  const int rows_per_thread = (n + (int) maxThreads - 1) / (int) maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      int base = r * n;
      for (int c = 0; c < n; ++c) flat[base + c] = row[c];
    }
  };

  for (unsigned t = 0; t < maxThreads; ++t) {
    int r0 = (int) t * rows_per_thread;
    int r1 = std::min(r0 + rows_per_thread, n);
    if (r0 >= r1) break;
    threads.emplace_back(worker, r0, r1);
  }
  for (auto &th: threads) th.join();

  return flat;
}

// ======================= PERFECT ============================
static void GeneratePerfListPerfectRowFlat(int m, int n,
                                           const Vector<int> &first_choices,
                                           FlatPreferenceList &flat) {
  int base = m * n;
  int first = first_choices[m];

  // rotation starting at "first"
  for (int i = 0; i < n; ++i) flat[base + i] = (first + i) % n;

  // shuffle tail [1..n-1]
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(flat.begin() + base + 1, flat.begin() + base + n, g);
}

static FlatPreferenceList GeneratePrefListsPerfectFlat(int n) {
  FlatPreferenceList flat(n * n);
  Vector<int> first(n);
  for (int i = 0; i < n; ++i) first[i] = i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(first.begin(), first.end(), g);

  std::vector<std::thread> threads;
  threads.reserve(n);
  for (int m = 0; m < n; ++m) {
    threads.emplace_back(GeneratePerfListPerfectRowFlat, m, n,
                         std::cref(first), std::ref(flat));
  }
  for (auto &th: threads) th.join();
  return flat;
}

// ======================= RANDOM (GROUPED) ===================
static void GeneratePrefListsRandomOneRowFlat(int n, int group_size,
                                              int row,
                                              const Vector<int> &ids,
                                              FlatPreferenceList &flat) {
  int base = row * n;
  int num_groups = (n + group_size - 1) / group_size;

  for (int g = 0; g < num_groups; ++g) {
    int start = g * group_size;
    int end = std::min(start + group_size, n);

    Vector<int> cur_ids(end - start);
    for (int i = start; i < end; ++i) cur_ids[i - start] = ids[i];

    std::mt19937 rng((uint32_t) std::random_device{}());
    std::shuffle(cur_ids.begin(), cur_ids.end(), rng);

    for (int i = start; i < end; ++i)
      flat[base + i] = cur_ids[i - start];
  }
}

static FlatPreferenceList GeneratePrefListsRandomFlat(int n, int group_size) {
  FlatPreferenceList flat(n * n);

  Vector<int> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = i; {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
  }

  const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
  const int rows_per_thread = (n + (int) maxThreads - 1) / (int) maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r)
      GeneratePrefListsRandomOneRowFlat(n, group_size, r, ids, flat);
  };

  for (unsigned t = 0; t < maxThreads; ++t) {
    int r0 = (int) t * rows_per_thread;
    int r1 = std::min(r0 + rows_per_thread, n);
    if (r0 >= r1) break;
    threads.emplace_back(worker, r0, r1);
  }
  for (auto &th: threads) th.join();

  return flat;
}

// ======================= SOLO (MEN) =========================
static void GeneratePrefListsManSoloRowFlat(int m, int n, FlatPreferenceList &flat) {
  int base = m * n;
  int original_m = m;
  if (original_m == n - 1) m = n - 2;

  int w1 = m;
  int w2 = (m - 1 + n - 1) % (n - 1);
  int w_last = n - 1;

  // anchors
  flat[base + 0] = w1;
  flat[base + (n - 2)] = w2;
  flat[base + (n - 1)] = w_last;

  int w = 0, rank = 1;
  while (rank < n - 2) {
    while (w == w1 || w == w2) ++w;
    flat[base + rank] = w;
    ++w;
    ++rank;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(flat.begin() + base + 1, flat.begin() + base + (n - 2), g);
}

static FlatPreferenceList GeneratePrefListsManSoloFlat(int n) {
  FlatPreferenceList flat(n * n);

  const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
  const int rows_per_thread = (n + (int) maxThreads - 1) / (int) maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r)
      GeneratePrefListsManSoloRowFlat(r, n, flat);
  };

  for (unsigned t = 0; t < maxThreads; ++t) {
    int r0 = (int) t * rows_per_thread;
    int r1 = std::min(r0 + rows_per_thread, n);
    if (r0 >= r1) break;
    threads.emplace_back(worker, r0, r1);
  }
  for (auto &th: threads) th.join();

  return flat;
}

// ======================= SOLO (WOMEN) =======================
static void GeneratePrefListsWomanSoloRowFlat(int w, int n, FlatPreferenceList &flat) {
  int base = w * n;

  if (w < n - 1) {
    int m1 = (w + 1) % (n - 1);
    int m2 = (m1 + n - 1) % n;

    flat[base + 0] = m1;
    flat[base + 1] = m2;

    int m = 0, rank = 2;
    while (rank < n) {
      while (m == m1 || m == m2) ++m;
      flat[base + rank] = m;
      ++m;
      ++rank;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(flat.begin() + base + 2, flat.begin() + base + n, g);
  } else {
    for (int rank = 0; rank < n; ++rank)
      flat[base + rank] = n - 1 - rank;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(flat.begin() + base, flat.begin() + base + n, g);
  }
}

static FlatPreferenceList GeneratePrefListsWomanSoloFlat(int n) {
  FlatPreferenceList flat(n * n);

  const int maxThreads = std::max(1u, std::thread::hardware_concurrency());
  const int rows_per_thread = (n + (int) maxThreads - 1) / (int) maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r)
      GeneratePrefListsWomanSoloRowFlat(r, n, flat);
  };

  for (unsigned t = 0; t < maxThreads; ++t) {
    int r0 = (int) t * rows_per_thread;
    int r1 = std::min(r0 + rows_per_thread, n);
    if (r0 >= r1) break;
    threads.emplace_back(worker, r0, r1);
  }
  for (auto &th: threads) th.join();

  return flat;
}

// ======================= TOP-LEVEL API ======================
inline void GenerateWorkload(WorkloadType type, int n,
                             FlatPreferenceList &flatM,
                             FlatPreferenceList &flatW,
                             int group_size = 5) {
  StopWatch sw;
  switch (type) {
    case RANDOM:
      flatM = GeneratePrefListsRandomFlat(n, group_size);
      flatW = GeneratePrefListsRandomFlat(n, group_size);
      break;
    case CONGESTED:
      flatM = GeneratePrefListsCongestedFlat(n);
      flatW = GeneratePrefListsCongestedFlat(n);
      break;
    case SOLO:
      flatM = GeneratePrefListsManSoloFlat(n);
      flatW = GeneratePrefListsWomanSoloFlat(n);
      break;
    case PERFECT:
      flatM = GeneratePrefListsPerfectFlat(n);
      flatW = GeneratePrefListsPerfectFlat(n);
      break;
    default:
      LOG(ERROR) << "Unknown workload type";
      break;
  }
  sw.Stop();
  LOG(INFO) << "Generated workload type: " << WorkloadTypeToString(type)
      << ", size: " << n
      << ", elapsed time: " << sw.GetEclapsedMs() << " ms.";
}

// ======================= SAVE / LOAD ========================
const String workloadDir = "data/workloads/";

inline String GetWorkloadFilename(WorkloadType type, size_t n, size_t group_size) {
  OutStringStream oss;
  oss << "smp_workload_";
  switch (type) {
    case RANDOM: oss << "random_" << n << "_" << group_size;
      break;
    case SOLO: oss << "solo_" << n;
      break;
    case CONGESTED: oss << "congested_" << n;
      break;
    case PERFECT: oss << "perfect_" << n;
      break;
  }
  oss << ".txt";
  return oss.str();
}

inline void SaveSmpWorkload(const String &filename,
                            const FlatPreferenceList &flatM,
                            const FlatPreferenceList &flatW,
                            int n) {
  const String fullpath = workloadDir + filename;
  OutFStream ofs(fullpath);
  ofs << n << "\n";

  // M
  for (int r = 0; r < n; ++r) {
    int base = r * n;
    for (int c = 0; c < n; ++c) ofs << flatM[base + c] << " ";
    ofs << "\n";
  }
  // W
  for (int r = 0; r < n; ++r) {
    int base = r * n;
    for (int c = 0; c < n; ++c) ofs << flatW[base + c] << " ";
    ofs << "\n";
  }
  ofs.close();
}

inline void LoadSmpWorkload(const String &filename,
                            FlatPreferenceList &flatM,
                            FlatPreferenceList &flatW,
                            int &n_out) {
  const String fullpath = workloadDir + filename;
  InFStream ifs(fullpath);
  if (!ifs.is_open()) {
    LOG(FATAL) << "Cannot open file: " << fullpath;
  }

  int n;
  ifs >> n;
  n_out = n;

  flatM.assign(n * n, 0);
  for (int r = 0; r < n; ++r) {
    int base = r * n;
    for (int c = 0; c < n; ++c) ifs >> flatM[base + c];
  }

  flatW.assign(n * n, 0);
  for (int r = 0; r < n; ++r) {
    int base = r * n;
    for (int c = 0; c < n; ++c) ifs >> flatW[base + c];
  }

  ifs.close();
}

// ======================= CACHED WRAPPER =====================
inline void GenerateWorkloadCached(WorkloadType type, int n,
                                   FlatPreferenceList &flatM,
                                   FlatPreferenceList &flatW,
                                   int group_size = 5) {
  MakeDirec(workloadDir);
  String filename = GetWorkloadFilename(type, n, group_size);

  if (FileExists(workloadDir + filename)) {
    LOG(INFO) << "Loaded existing dataset from " << filename;
    int n_file = 0;
    LoadSmpWorkload(filename, flatM, flatW, n_file);
    if (n_file != n) {
      LOG(WARNING) << "Loaded n(" << n_file << ") != requested n(" << n
          << "), using loaded size.";
    }
  } else {
    LOG(INFO) << "Dataset not found, generating new one.";
    GenerateWorkload(type, n, flatM, flatW, group_size);
    SaveSmpWorkload(filename, flatM, flatW, n);
    LOG(INFO) << "Saved generated dataset to " << filename;
  }
}


#endif //GENERATE_WORKLOADS_FLAT_H
