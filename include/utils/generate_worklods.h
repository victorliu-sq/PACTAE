#ifndef GENERATE_WORKLODS_H
#define GENERATE_WORKLODS_H

#include <algorithm>
#include <random>
#include "stopwatch.h"
#include "types.h"
#include <filesystem>
#include <thread>

// -------------------- Congested Case ----------------------------
// ----------------------------------------------------------------
// static PreferenceLists GeneratePrefListsCongested(int n) {
//   PreferenceLists pls(n, std::vector<int>(n, 0));
//
//   // Using random_shuffle for C++98 compatibility
//   std::srand(static_cast<unsigned int>(getNanoSecond()));
//
//   // (2) For each row: if i != j, randomly assign all numbers from 0 to n - 1 to
//   // its left values
//   std::vector<int> row_values(n);
//   for (int i = 0; i < n; i++) {
//     // printf("Current Row: %d\n", i);
//     row_values[i] = i;
//   }
//
//   std::random_shuffle(row_values.begin(), row_values.end());
//
//   for (int i = 0; i < n; ++i) {
//     pls[i] = row_values;
//   }
//
//   return pls;
// }
static PreferenceLists GeneratePrefListsCongested(int n) {
  PreferenceLists pls(n, std::vector<int>(n, 0));

  std::srand(static_cast<unsigned int>(getNanoSecond()));

  std::vector<int> row_values(n);
  for (int i = 0; i < n; ++i) {
    row_values[i] = i;
  }
  std::random_shuffle(row_values.begin(), row_values.end());

  const int maxThreads = std::thread::hardware_concurrency();
  const int rows_per_thread = (n + maxThreads - 1) / maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
      pls[row] = row_values; // Assigning pre-shuffled row_values
    }
  };

  for (int thread_id = 0; thread_id < maxThreads; ++thread_id) {
    int start_row = thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, n);
    if (start_row >= end_row) break; // Avoid extra threads
    threads.emplace_back(worker, start_row, end_row);
  }

  for (auto &thread: threads) {
    thread.join();
  }

  return pls;
}

// --------------------------- Perfect Case -------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePerfListPerfectRow(int m,
                                       int n,
                                       std::vector<int> first_choices,
                                       PreferenceLists &pls) {
  int first_choice = first_choices[m];
  for (int i = 0; i < n; i++) {
    pls[m][i] = (first_choice + i) % n;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(pls[m].begin() + 1, pls[m].end(), g);
}

// Function to manage threads and return the preference list matrix
static PreferenceLists GeneratePrefListsPerfect(int n) {
  PreferenceLists pls(n, std::vector<int>(n));

  std::vector<std::thread> threads;
  std::vector<int> first_choices(n, 0);

  for (int m = 0; m < n; m++) {
    first_choices[m] = m;
  }

  std::random_device rd; // Seed
  std::mt19937 g(rd()); // Mersenne Twister engine

  // Shuffle the vector
  std::shuffle(first_choices.begin(), first_choices.end(), g);
  // Launch threads
  for (int m = 0; m < n; m++) {
    threads.emplace_back(GeneratePerfListPerfectRow, m, n, first_choices,
                         std::ref(pls));
  }

  // Join threads
  for (auto &th: threads) {
    th.join();
  }

  return pls;
}


// -------------------- Random Case ----------------------------
// ----------------------------------------------------------------
static void GeneratePrefListsRandomOneRow(const int n, const int group_size,
                                          std::vector<int> &pl,
                                          std::vector<int> &ids) {
  int num_groups = (n + group_size - 1) / group_size;
  int cur_group_size;
  for (int group_id = 0; group_id < num_groups; group_id++) {
    if (group_id == num_groups - 1 && n % group_size != 0) {
      cur_group_size = n % group_size;
    } else {
      cur_group_size = group_size;
    }
    std::vector<int> cur_ids(cur_group_size, 0);
    for (int i = 0; i < cur_ids.size(); i++) {
      cur_ids[i] = ids[group_id * group_size + i];
    }

    std::mt19937 rng(static_cast<unsigned int>(getNanoSecond()));
    std::shuffle(cur_ids.begin(), cur_ids.end(), rng);

    for (int i = 0; i < cur_ids.size(); i++) {
      pl[group_id * group_size + i] = cur_ids[i];
    }
  }
}

// static PreferenceLists GeneratePrefListsRandom(const int n,
//                                                const int group_size) {
//   PreferenceLists pls(n, std::vector<int>(n));
//   std::vector<int> ids(n);
//   for (int i = 0; i < n; i++) {
//     ids[i] = i;
//   }
//   std::srand(static_cast<unsigned int>(getNanoSecond()));
//   std::random_shuffle(ids.begin(), ids.end());
//   int maxThreads =
//       std::thread::hardware_concurrency(); // Use hardware concurrency as a
//   // limit.
//   std::vector<std::thread> threads;
//   threads.reserve(maxThreads);
//
//   for (int i = 0; i < n; ++i) {
//     if (threads.size() >= maxThreads) {
//       // Wait for all the threads in this batch to complete.
//       for (auto &th: threads) {
//         th.join();
//       }
//       threads.clear();
//     }
//     threads.emplace_back(GeneratePrefListsRandomOneRow, n, group_size,
//                          std::ref(pls[i]), std::ref(ids));
//   }
//
//   // Ensure any remaining threads are joined.
//   for (auto &th: threads) {
//     th.join();
//   }
//
//   return pls;
// }
static PreferenceLists GeneratePrefListsRandom(const int n, const int group_size) {
  PreferenceLists pls(n, std::vector<int>(n));
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i) {
    ids[i] = i;
  }

  std::srand(static_cast<unsigned int>(getNanoSecond()));
  std::random_shuffle(ids.begin(), ids.end());

  const int maxThreads = std::thread::hardware_concurrency();
  const int rows_per_thread = (n + maxThreads - 1) / maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
      GeneratePrefListsRandomOneRow(n, group_size, pls[row], ids);
    }
  };

  for (int thread_id = 0; thread_id < maxThreads; ++thread_id) {
    int start_row = thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, n);
    if (start_row >= end_row) break; // Avoid launching unnecessary threads
    threads.emplace_back(worker, start_row, end_row);
  }

  for (auto &thread: threads) {
    thread.join();
  }

  return pls;
}

// --------------------------- Solo Case --------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePrefListsManSoloRow(int m, int n, PreferenceLists &plM) {
  std::vector<int> preference_list(n);
  int original_m = m;
  if (original_m == n - 1) {
    m = n - 2;
  }
  int w1 = m, w2 = (m - 1 + n - 1) % (n - 1), w_last = n - 1;
  preference_list[0] = w1;
  preference_list[n - 2] = w2;
  preference_list[n - 1] = w_last;
  int w = 0, rank = 1;
  while (rank < n - 2) {
    while (w == w1 || w == w2) {
      w++;
    }
    preference_list[rank] = w;
    w++;
    rank++;
  }
  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 1, preference_list.begin() + n - 2, g);

  plM[original_m] = preference_list;
}

// Function to manage threads and return the preference list matrix
// static PreferenceLists GeneratePrefListsManSolo(int n) {
//   PreferenceLists plM(n);
//
//   int max_threads = std::thread::hardware_concurrency();
//   std::vector<std::thread> threads;
//   threads.reserve(max_threads);
//
//   // Launch threads in batches
//   for (int m = 0; m < n; m++) {
//     if (threads.size() >= max_threads) {
//       for (auto &th: threads) th.join();
//       threads.clear();
//     }
//
//     threads.emplace_back(GeneratePrefListsManSoloRow, m, n, std::ref(plM));
//   }
//
//   // Join remaining threads
//   for (auto &th: threads) th.join();
//
//   return plM;
// }
static PreferenceLists GeneratePrefListsManSolo(int n) {
  PreferenceLists plM(n);

  const int maxThreads = std::thread::hardware_concurrency();
  const int rows_per_thread = (n + maxThreads - 1) / maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
      GeneratePrefListsManSoloRow(row, n, plM);
    }
  };

  for (int thread_id = 0; thread_id < maxThreads; ++thread_id) {
    int start_row = thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, n);
    if (start_row >= end_row) break; // avoid unnecessary threads
    threads.emplace_back(worker, start_row, end_row);
  }

  for (auto &thread: threads) {
    thread.join();
  }

  return plM;
}

static void GeneratePrefListsWomanSoloRow(int m, int n,
                                          std::vector<std::vector<int> > &plW) {
  std::vector<int> preference_list(n);
  if (m < n - 1) {
    int w1 = (m + 1) % (n - 1), w2 = (w1 + n - 1) % n;
    preference_list[0] = w1;
    preference_list[1] = w2;
    int w = 0, rank = 2;
    while (rank < n) {
      while (w == w1 || w == w2) {
        w++;
      }
      preference_list[rank] = w;
      w++;
      rank++;
    }
  } else {
    for (int rank = 0; rank < n - 1; rank++) {
      preference_list[rank] = n - 1 - rank;
    }
  }

  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 2, preference_list.end(), g);
  plW[m] = preference_list;
}

// static PreferenceLists GeneratePrefListsWomanSolo(int n) {
//   PreferenceLists plW(n);
//
//   int max_threads = std::thread::hardware_concurrency();
//   std::vector<std::thread> threads;
//   threads.reserve(max_threads);
//
//   // Launch threads in batches
//   for (int m = 0; m < n; m++) {
//     if (threads.size() >= max_threads) {
//       for (auto &th: threads) th.join();
//       threads.clear();
//     }
//
//     threads.emplace_back(GeneratePrefListsWomanSoloRow, m, n, std::ref(plW));
//   }
//
//   // Join remaining threads
//   for (auto &th: threads) th.join();
//
//   return plW;
// }
static PreferenceLists GeneratePrefListsWomanSolo(int n) {
  PreferenceLists plW(n);

  const int maxThreads = std::thread::hardware_concurrency();
  const int rows_per_thread = (n + maxThreads - 1) / maxThreads;

  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  auto worker = [&](int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
      GeneratePrefListsWomanSoloRow(row, n, plW);
    }
  };

  for (int thread_id = 0; thread_id < maxThreads; ++thread_id) {
    int start_row = thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, n);
    if (start_row >= end_row) break; // avoid unnecessary threads
    threads.emplace_back(worker, start_row, end_row);
  }

  for (auto &thread: threads) {
    thread.join();
  }

  return plW;
}

// --------------------------- Solo Case --------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePrefListsManSoloRandRow(int m, int n,
                                            std::vector<std::vector<int> > &plM,
                                            std::vector<int> men_map,
                                            std::vector<int> women_map) {
  std::vector<int> preference_list(n);
  int original_m = m;

  if (original_m == n - 1) {
    m = n - 2;
  }
  int w1 = m, w2 = (m - 1 + n - 1) % (n - 1), w_last = n - 1;
  // preference_list[0] = w1;
  // preference_list[n - 2] = w2;
  // preference_list[n - 1] = w_last;
  preference_list[0] = women_map[w1];
  preference_list[n - 2] = women_map[w2];
  preference_list[n - 1] = women_map[w_last];
  int w = 0, rank = 1;
  while (rank < n - 2) {
    while (w == w1 || w == w2) {
      w++;
    }
    // randomized label of woman
    // preference_list[rank] = w;
    preference_list[rank] = women_map[w];
    w++;
    rank++;
  }
  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 1, preference_list.begin() + n - 2, g);

  // randomized label of man
  int mapped_m = men_map[original_m];
  // plM[original_m] = preference_list;
  plM[mapped_m] = preference_list;
}

static void GeneratePrefListsWomanSoloRandRow(
  int woman, int n, std::vector<std::vector<int> > &plW,
  std::vector<int> men_map, std::vector<int> women_map) {
  // std::cout << "Init PrefLists for woman " << woman << std::endl;
  std::vector<int> preference_list(n);
  if (woman < n - 1) {
    int m1 = (woman + 1) % (n - 1), m2 = (m1 + n - 1) % n;
    preference_list[0] = men_map[m1];
    preference_list[1] = men_map[m2];
    int m = 0, rank = 2;
    while (rank < n) {
      while (m == m1 || m == m2) {
        m++;
      }
      preference_list[rank] = men_map[m];
      m++;
      rank++;
    }
  } else {
    for (int rank = 0; rank < n - 1; rank++) {
      preference_list[rank] = rank;
    }
  }

  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  if (woman < n - 1) {
    std::shuffle(preference_list.begin() + 2, preference_list.end(), g);
  } else {
    std::shuffle(preference_list.begin(), preference_list.end(), g);
  }
  // plW[woman] = preference_list;
  plW[women_map[woman]] = preference_list;
}

static void GenerateRandomizedLabelPreflistsSolo(int n, PreferenceLists &plM,
                                                 PreferenceLists &plW) {
  plM.resize(n);
  plW.resize(n);
  std::vector<int> men_map(n, 0);
  std::vector<int> women_map(n, 0);

  std::vector<std::thread> threadsM;
  std::vector<std::thread> threadsW;

  // Fill the vector with numbers from 0 to n-1
  for (int i = 0; i < n; ++i) {
    women_map[i] = i;
    men_map[i] = i;
  }

  // Obtain a random number generator
  std::random_device rd; // Seed
  std::mt19937 g(rd()); // Mersenne Twister engine

  // Shuffle the vector
  std::shuffle(men_map.begin(), men_map.end(), g);
  std::shuffle(women_map.begin(), women_map.end(), g);

  // Print the shuffled vector
  // std::cout << "Shuffled Men: ";
  // for (const auto &man : men_map) {
  //   std::cout << man << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "Shuffled Women: ";
  // for (const auto &woman : women_map) {
  //   std::cout << woman << " ";
  // }

  // std::cout << std::endl;

  // std::cout << "Generate PLM: ";
  // Launch threads for Men preferences
  for (int m = 0; m < n; m++) {
    threadsM.emplace_back(GeneratePrefListsManSoloRandRow, m, n, std::ref(plM),
                          men_map, women_map);
  }

  // Join threads
  for (auto &th: threadsM) {
    th.join();
  }

  // Clear the threads vector before launching new threads
  // Otherwise, you get segfaults
  // threads.clear();

  // std::cout << "Generate PLW: ";
  // Launch threads for Women preferences
  for (int w = 0; w < n; w++) {
    threadsW.emplace_back(GeneratePrefListsWomanSoloRandRow, w, n,
                          std::ref(plW), men_map, women_map);
  }

  // Join threads
  for (auto &th: threadsW) {
    th.join();
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// --------------------------- Generate Workloads -------------------------------
// ------------------------------------------------------------------------
inline void GenerateWorkload(WorkloadType type, int n, PreferenceLists &plM, PreferenceLists &plW, int group_size = 5) {
  StopWatch sw;
  switch (type) {
    case RANDOM:
      plM = GeneratePrefListsRandom(n, group_size);
      plW = GeneratePrefListsRandom(n, group_size);
      break;
    case CONGESTED:
      plM = GeneratePrefListsCongested(n);
      plW = GeneratePrefListsCongested(n);
      break;
    case SOLO:
      plM = GeneratePrefListsManSolo(n);
      plW = GeneratePrefListsWomanSolo(n);
      break;
    case PERFECT:
      plM = GeneratePrefListsPerfect(n);
      plW = GeneratePrefListsPerfect(n);
      break;
    default:
      LOG(ERROR) << "Unknown workload type";
      break;
  }
  sw.Stop();

  std::string workload_type_str;
  switch (type) {
    case PERFECT:
      workload_type_str = "RANDOM";
      break;
    case RANDOM:
      workload_type_str = "RANDOM";
      break;
    case CONGESTED:
      workload_type_str = "CONGESTED";
      break;
    case SOLO:
      workload_type_str = "SOLO";
      break;
    default:
      workload_type_str = "UNKNOWN";
      break;
  }
  LOG(INFO) << "Generated workload type: " << workload_type_str
      << ", size: " << n
      << ", elapsed time: " << sw.GetEclapsedMs() << " ms.";
}


// --------------------------- Save and Load Workloads --------------------
// ------------------------------------------------------------------------
// const String workloadDir = "data/workloads/";
//
// inline String GetWorkloadFilename(WorkloadType type, size_t n, size_t group_size) {
//   OutStringStream oss;
//   oss << "smp_workload_";
//   switch (type) {
//     case RANDOM:
//       oss << "random_" << n << "_" << group_size;
//       break;
//     case SOLO:
//       oss << "solo_" << n;
//       break;
//     case CONGESTED:
//       oss << "congested_" << n;
//       break;
//     case PERFECT:
//       oss << "perfect_" << n;
//       break;
//   }
//
//   oss << ".txt";
//   return oss.str();
// }


inline void SaveSmpWorkload(const String &filename,
                            const PreferenceLists &plM,
                            const PreferenceLists &plW) {
  const String fullpath = workloadDir + filename;
  OutFStream ofs(fullpath);
  size_t n = plM.size();

  ofs << n << "\n";

  for (const auto &pl: plM) {
    for (int v: pl) {
      ofs << v << " ";
    }
    ofs << "\n";
  }

  // separate plM and plW
  // ofs << "\n";

  for (const auto &pl: plW) {
    for (int v: pl) {
      ofs << v << " ";
    }
    ofs << "\n";
  }

  ofs.close();
}

inline void LoadSmpWorkload(const String &filename,
                            PreferenceLists &plM,
                            PreferenceLists &plW) {
  const String fullpath = workloadDir + filename;
  InFStream ifs(fullpath);
  if (!ifs.is_open()) {
    LOG(FATAL) << "Cannot open file: " << fullpath;
  }

  size_t n;
  ifs >> n;

  plM.resize(n);
  for (auto &pl: plM) {
    pl.resize(n);
    for (int i = 0; i < n; ++i) {
      ifs >> pl[i];
    }
  }

  plW.resize(n);
  for (auto &pl: plW) {
    pl.resize(n);
    for (int i = 0; i < n; ++i) {
      ifs >> pl[i];
    }
  }

  ifs.close();
}


inline void GenerateWorkloadCached(WorkloadType type, int n, PreferenceLists &plM, PreferenceLists &plW,
                                   int group_size = 5) {
  // By default, all Smp Data
  // MakeDirec("data/workloads/");
  MakeDirec(workloadDir);
  String filename = GetWorkloadFilename(type, n, group_size);

  // Bug fixed: missing path seprator '/' in the end
  if (FileExists(workloadDir + filename)) {
    LOG(INFO) << "Loaded existing dataset from " << filename;
    LoadSmpWorkload(filename, plM, plW);
  } else {
    LOG(INFO) << "Dataset not found, generating new one.";
    GenerateWorkload(type, n, plM, plW, group_size);
    SaveSmpWorkload(filename, plM, plW);
    LOG(INFO) << "Saved generated dataset to " << filename;
  }
}

#endif //GENERATE_WORKLODS_H
