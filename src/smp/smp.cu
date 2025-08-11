#include "utils/utils.h"
#include <fstream>
#include <iostream>
#include <smp/smp.h>
#include <sstream>
#include <vector>

namespace bamboosmp {
  auto SmpObj::CreateFromPrefLists(const PreferenceLists &pl_m,
                                   const PreferenceLists &pl_w,
                                   int n) -> UPtr<SmpObj> {
    return UPtr<SmpObj>(new SmpObj(pl_m, pl_w, n));
  }

  SmpObj::SmpObj(const PreferenceLists &pl_m, const PreferenceLists &pl_w, int n)
    : flatten_pref_lists_m_vec(ParallelFlattenHost(pl_m, n)),
      flatten_pref_lists_w_vec(ParallelFlattenHost(pl_w, n)) {
    // Crucial pointer initialization step
    this->InitPointers();
  }


  void SmpObj::FlattenRowHost(const PreferenceLists &pls, Vector<int> &flat_pl, int row, int n) {
    for (int col = 0; col < n; col++) {
      flat_pl[IDX_MUL_ADD(row, n, col)] = pls[row][col];
    }
  }

  auto SmpObj::ParallelFlattenHost(const PreferenceLists &pl, int n) -> Vector<int> {
    Vector<int> flat_pl(SIZE_MUL(n, n));

    int num_threads = std::thread::hardware_concurrency(); // Get the number of
    // supported threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < n; ++i) {
      if (threads.size() >= num_threads) {
        // Wait for all the threads in this batch to complete.
        for (auto &th: threads) {
          th.join();
        }
        threads.clear();
      }

      threads.emplace_back(FlattenRowHost, std::ref(pl), std::ref(flat_pl), i, n);
    }

    // Join the threads
    for (auto &th: threads) {
      th.join();
    }

    return flat_pl;
  }
} // namespace bamboosmp
