#include <set>
#include <smp/smp_engine_la_old.cuh>
#include <utils/utils.h>
#include "utils/launcher.h"
#include <vector>
#include <glog/logging.h>
#include <cuda.h>
#include "utils/array_view.h"

namespace bamboosmp {
  SmpEngineLaOld::SmpEngineLaOld(const SmpObj &smp, int size)
    : smp_(smp), n_(size) {
    cudaSetDevice(0);
    stable_matching_.resize(n_, 0);
    std::set<int> topChoicesSet;
    for (int m = 0; m < n_; ++m) {
      // int topChoice = smp_.flatten_pref_lists_m_[m * n_];
      int topChoice = smp_.flatten_pref_lists_m_vec[IDX_MUL(m, n_)];
      topChoicesSet.insert(topChoice);
      stable_matching_[m] = topChoice;
    }
    is_perfect_ = topChoicesSet.size() == n_;
  }

  auto SmpEngineLaOld::GetStableMarriage() const -> std::vector<int> {
    std::vector<int> result(n_);
    for (int i = 0; i < n_; i++) {
      result[i] = stable_matching_[i];
    }
    return result;
  }

  void SmpEngineLaOld::PrintMatching() const {
    LOG(INFO) << "Stable Matching:";
    for (int i = 0; i < n_; ++i) {
      LOG(INFO) << "(Man:" << i << " is paired with Woman:" << stable_matching_[i] << ")";
    }
  }

  void SmpEngineLaOld::PostProc() {
    LOG(INFO) << "SmpEngineLa: PostProc Starts";
    for (int w = 0; w < n_; w++) {
      int m = pref_lists_w_view_[IDX_MUL_ADD(w, n_, host_partner_rank_[w])];
      stable_matching_[m] = w;
    }

    // for (int w = 0; w < n_; w++) {
    //   int m = pref_lists_w_view_[IDX_MUL_ADD(w, n_, atomic_partner_rank_[w].load())];
    //   stable_matching_[m] = w;
    // }

    LOG(INFO) << "SmpEngineLa: PostProc Completes";
  }

  /* ------------------------------------- Init ------------------------------------------------ */
  /* ------------------------------------------------------------------------------------------------ */
  void SmpEngineLaOld::InitProc() {
    const size_t arr_size = static_cast<size_t>(n_);
    const size_t mtx_size = SIZE_MUL(n_, n_);

    host_partner_rank_.assign(arr_size, n_);
    host_next_proposed_w_.assign(arr_size, 0);

    host_rank_mtx_w_.resize(mtx_size);
    device_partner_rank_.resize(arr_size);
    device_next_proposed_w_.resize(arr_size);
    device_pref_lists_m_.resize(mtx_size);
    device_pref_lists_w_.resize(mtx_size);
    device_rank_mtx_w_.resize(mtx_size);
    device_prnodes_m_.resize(mtx_size);

    CUDA_CHECK(cudaMallocHost(&host_prnodes_m_ptr_, mtx_size * sizeof(PRNode), cudaHostAllocDefault));

    thrust::copy(host_partner_rank_.begin(),
                 host_partner_rank_.end(),
                 device_partner_rank_.begin());

    thrust::copy(host_next_proposed_w_.begin(),
                 host_next_proposed_w_.end(),
                 device_next_proposed_w_.begin());

    thrust::copy(smp_.flatten_pref_lists_m_vec.begin(),
                 smp_.flatten_pref_lists_m_vec.end(),
                 device_pref_lists_m_.begin());

    thrust::copy(smp_.flatten_pref_lists_w_vec.begin(),
                 smp_.flatten_pref_lists_w_vec.end(),
                 device_pref_lists_w_.begin());

    pref_lists_w_view_ = RawPtr(smp_.flatten_pref_lists_w_vec);

    LaunchKernelForEach(default_stream_, mtx_size,
                        InitRankMatrixKernelTasklet{
                          n_,
                          ConstArrayView<int>(device_pref_lists_w_),
                          ArrayView<int>(device_rank_mtx_w_)
                        });

    auto start_init_prmtx = getNanoSecond();
    LaunchKernelForEach(default_stream_, mtx_size,
                        InitPRMatrixKernelTasklet{
                          n_,
                          ConstArrayView<int>(device_rank_mtx_w_),
                          ConstArrayView<int>(device_pref_lists_m_),
                          ArrayView<PRNode>(device_prnodes_m_)
                        });

    CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_ptr_,
      RawPtr(device_prnodes_m_),
      SIZE_MUL(n_, n_) * sizeof(PRNode),
      cudaMemcpyDeviceToHost,
      monitor_stream_.cuda_stream()));
    monitor_stream_.Sync();

    auto end_init_prmtx = getNanoSecond();
    std::cout << "SmpEngineLa::InitPRMtx: " << (end_init_prmtx - start_init_prmtx) / 1e6 << std::endl;

    // Parallel LAExecution
    atomic_partner_rank_ = new std::atomic<int>[n_];
    for (int i = 0; i < n_; i++) {
      atomic_partner_rank_[i].store(n_);
    }
    LOG(INFO) << "Initialization procedure completed.";
  }

  /* ------------------------------------- Core Computation ------------------------------------- */

  void SmpEngineLaOld::CoreProc() {
    auto start_time = getNanoSecond();
    // LA-Seq-CPU
    PROFILE_SCOPE("LA-Seq-CPU");
    for (int m = 0; m < n_; m++) {
      LAProcedure(m);
    }

    // LA-Par-CPU
    // PROFILE_SCOPE("LA-Par-CPU");
    // std::vector<std::thread> threads;
    // int max_num_threads = std::min(std::thread::hardware_concurrency(), 96u);
    // threads.reserve(max_num_threads);
    // // for (int m = 0; m < n_; m++) {
    // //   LAParallelProcedure(m);
    // // }
    //
    // for (int tid = 0; tid < n_; ++tid) {
    //   if (threads.size() >= max_num_threads) {
    //     // Wait for current batch to finish
    //     for (auto &th: threads) {
    //       th.join();
    //     }
    //     threads.clear();
    //   }
    //   threads.emplace_back(&SmpEngineLa::LAParallelProcedure, this, tid);
    // }
    //
    // for (auto &thread: threads) {
    //   thread.join();
    // }
    //
    auto end_time = getNanoSecond();
    std::cout << "Exec Phase - LA is : " << (end_time - start_time) / 1e6 << " ms" << std::endl;
  }

  void SmpEngineLaOld::LAProcedure(int m) {
    auto host_partner_rank_ptr = host_partner_rank_.data();
    auto host_next_proposed_w_ptr = host_next_proposed_w_.data();
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    // printf("run LA Procedure on man %d\n", m);
    m_idx = m;
    w_rank = 0;
    PRNode temp_node;
    bool is_matched = false;
    int iterations = 0;
    while (!is_matched) {
      iterations += 1;
      temp_node = host_prnodes_m_ptr_[m_idx * n_ + w_rank];
      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      // printf("man %d proposes to rank %d-th woman %d, ", m_idx, w_rank, w_idx);
      p_rank = host_partner_rank_ptr[w_idx];
      if (p_rank == n_) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_ptr[w_idx] = m_rank;
        is_matched = true;
        // printf("and woman %d is unpaired, so STOP\n", w_idx);
      } else if (p_rank > m_rank) {
        // std::cout << "man " << m << " gets rejects (1)" << std::endl;
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_ptr[w_idx] = m_rank;

        // m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
        m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
        w_rank = host_next_proposed_w_ptr[m_idx];
        // printf("and gets ACCEPTED. Since woman %d is paired with man %d, so "
        //        "SWITCH\n",
        //        w_idx, m_idx);
      } else {
        w_rank++;
        // std::cout << "man " << m << " gets rejects (2)" << std::endl;
        // printf("and gets REJECTED, so move to next woman\n");
      }
    }
  }

  // void SmpEngineLa::LAParallelProcedure(int m) {
  //   // auto host_partner_rank_ptr = host_partner_rank_.data();
  //   auto host_next_proposed_w_ptr = host_next_proposed_w_.data();
  //   int w_idx, m_rank, m_idx, w_rank, p_rank;
  //   m_idx = m;
  //   w_rank = 0;
  //
  //   bool done = false;
  //   while (!done) {
  //     // Prefetch ahead aggressively (prefetch next few entries)
  //     // constexpr int PREFETCH_DISTANCE = 4;
  //     // int prefetch_idx = m_idx * n_ + w_rank + PREFETCH_DISTANCE;
  //     // if (prefetch_idx < n_ * n_) {
  //     //   // bounds check
  //     //   __builtin_prefetch(&host_prnodes_m_ptr_[prefetch_idx], 0, 1);
  //     // }
  //     PRNode temp_node = host_prnodes_m_ptr_[m_idx * n_ + w_rank];
  //     w_idx = temp_node.idx_;
  //     m_rank = temp_node.rank_;
  //     w_rank += 1;
  //     p_rank = atomic_partner_rank_[w_idx].load();
  //
  //     if (p_rank < m_rank) {
  //       continue;
  //     }
  //
  //     bool cas_success = false;
  //     while (!cas_success && m_rank < p_rank) {
  //       int expected = p_rank;
  //       if (!atomic_partner_rank_[w_idx].compare_exchange_strong(expected,
  //                                                                m_rank)) {
  //         p_rank = expected;
  //       } else {
  //         cas_success = true;
  //         host_next_proposed_w_ptr[m_idx] = w_rank;
  //         if (p_rank != n_) {
  //           // m_idx = smp_.flatten_pref_lists_w_[IDX_MUL_ADD(w_idx, n_, p_rank)];
  //           m_idx = pref_lists_w_view_[IDX_MUL_ADD(w_idx, n_, p_rank)];
  //           w_rank = host_next_proposed_w_ptr[m_idx];
  //         } else {
  //           done = true;
  //         }
  //       }
  //     }
  //   }
  // }

  void SmpEngineLaOld::LAParallelProcedure(int m) {
    auto host_next_proposed_w_ptr = host_next_proposed_w_.data();

    int m_idx = m;
    int w_rank = 0;
    bool done = false;

    // constexpr int PREFETCH_DISTANCE = 4; // tuned distance
    // const int total_nodes = n_ * n_; // total size for bounds check

    while (!done) {
      int current_idx = m_idx * n_ + w_rank;

      // Explicit prefetch next entries (tuned prefetch distance)
      // int prefetch_idx = current_idx + PREFETCH_DISTANCE;
      // if (prefetch_idx < total_nodes) {
      // __builtin_prefetch(&(host_prnodes_m_ptr_[prefetch_idx]), 0, 1);
      // }

      // Prevent compiler from optimizing away prefetch
      // asm volatile("" ::: "memory");

      PRNode temp_node = host_prnodes_m_ptr_[current_idx];

      int w_idx = temp_node.idx_;
      int m_rank = temp_node.rank_;
      w_rank += 1;

      // Load atomic value with relaxed order first to reduce contention
      int p_rank = atomic_partner_rank_[w_idx].load(std::memory_order_relaxed);

      if (p_rank < m_rank) {
        continue;
      }

      bool cas_success = false;
      while (!cas_success && m_rank < p_rank) {
        int expected = p_rank;
        if (!atomic_partner_rank_[w_idx].compare_exchange_strong(expected,
                                                                 m_rank,
                                                                 std::memory_order_acq_rel)) {
          p_rank = expected;
        } else {
          cas_success = true;
          host_next_proposed_w_ptr[m_idx] = w_rank;

          if (p_rank != n_) {
            // Fetch next indices
            int next_m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
            int next_w_rank = host_next_proposed_w_ptr[next_m_idx];

            // Prefetch upcoming data aggressively for the next loop iteration
            // prefetch_idx = next_m_idx * n_ + next_w_rank;
            // if (prefetch_idx < total_nodes) {
            //   __builtin_prefetch(&(host_prnodes_m_ptr_[prefetch_idx]), 0, 1);
            // }

            m_idx = next_m_idx;
            w_rank = next_w_rank;
          } else {
            done = true;
          }
        }
      }
    }
  }
} // namespace bamboosmp
