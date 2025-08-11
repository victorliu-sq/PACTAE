#include <cstdint>
#include <iostream>
#include <mutex>
#include <smp/gs.cuh>
#include <thread>
#include <utils/utils.h>
#include <vector>
#include <utils/timer.h>

namespace bamboosmp {
  auto GS::StartGSParallel() -> float {
    PROFILE_SCOPE("Total Time Gs-Par-Cpu");
    auto start_time = getNanoSecond();
    InitCpuPar();

    printf("StartGSParallel Exec\n");
    // std::vector<std::thread> threads;
    // for (int i = 0; i < n_; ++i) {
    //   threads.emplace_back(&GS::GSParallelProcedure, this, i);
    // }
    //
    // auto intermediate_time = getNanoSecond();
    // printf("Time to emplace all threads is %f\n",
    //        (intermediate_time - start_time) / 1e6);
    //
    // for (auto &thread: threads) {
    //   thread.join();
    // }
    std::vector<std::thread> threads;
    threads.reserve(max_num_threads_);

    // Launch threads in batches, each handling one distinct woman (0 to n_-1)
    for (int tid = 0; tid < max_num_threads_; ++tid) {
      threads.emplace_back(&GS::GSParallelProcedure, this, tid);
    }

    for (auto &thread: threads) {
      thread.join();
    }

    auto end_time = getNanoSecond();
    std::cout << "Exec Phase-GS-Par-CPU is: " << (end_time - start_time) / 1e6 << std::endl;
    return (end_time - start_time) / 1e6;
  }

  void GS::GSParallelProcedure(int tid) {
    // printf("Thread ID is %d\n", tid);
    const int *pref_lists_m = RawPtr(smp_.flatten_pref_lists_m_vec);
    const int *pref_lists_w = RawPtr(smp_.flatten_pref_lists_w_vec);

    int w_idx, m_rank, m_idx, w_rank, p_rank;
    auto free_men_queue = free_men_queues_[tid];

    m_idx = free_men_queue.front();
    free_men_queue.pop();
    w_rank = 0;

    bool done = false;
    while (!done) {
      // w_idx = smp_.flatten_pref_lists_m_[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // w_idx = pref_lists_m[m_idx * n_ + w_rank];
      w_idx = pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // m_rank = host_rank_mtx_w_[w_idx * n_ + m_idx];
      m_rank = host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

      // cout_mutex.lock();
      // std::cout << "[Thread " << tid << "] Man " << m_idx << " proposed to
      // rank-"
      //           << w_rank << " Woman " << w_idx << std::endl;
      // cout_mutex.unlock();
      w_rank += 1;

      p_rank = atomic_partner_rank_[w_idx].load();

      if (p_rank < m_rank) {
        // cout_mutex.lock();
        // std::cout << "[Thread " << tid << "] Man " << m_idx
        //           << " is rejected by his rank-" << w_rank << " Woman " <<
        //           w_idx
        //           << std::endl;
        // cout_mutex.unlock();
        continue;
      }

      // cout_mutex.lock();
      // std::cout << "[Thread " << tid << "] Queue size " <<
      // free_men_queue.size()
      //           << std::endl;
      // cout_mutex.unlock();

      while (m_rank < p_rank) {
        int expected = p_rank;
        if (atomic_partner_rank_[w_idx].compare_exchange_strong(expected,
                                                                m_rank)) {
          // cout_mutex.lock();
          // std::cout << "[Thread " << tid << "] Man " << m_idx
          //           << " is accepted by his rank-" << w_rank << " Woman " <<
          //           w_idx
          //           << std::endl;
          // cout_mutex.unlock();
          next_proposed_w_[m_idx] = w_rank;
          if (p_rank != n_) {
            // int new_free_man = smp_.flatten_pref_lists_w_[IDX_MUL_ADD(w_idx, n_, p_rank)];
            // int new_free_man = pref_lists_w[w_idx * n_ + p_rank];
            int new_free_man = pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
            free_men_queue.push(new_free_man);
            // cout_mutex.lock();
            // std::cout << "[Thread " << tid << "] Man " << new_free_man
            //           << " becomes free again" << std::endl;
            // cout_mutex.unlock();
          }
          if (free_men_queue.size() > 0) {
            // std::cout << "[Thread " << tid << "] Queue size "
            //           << free_men_queue.size() << std::endl;
            m_idx = free_men_queue.front();

            // cout_mutex.lock();
            // std::cout << "[Thread " << tid << "] pop out a Man " << m_idx
            //           << std::endl;

            // cout_mutex.unlock();
            free_men_queue.pop();
            w_rank = next_proposed_w_[m_idx];
          } else {
            done = true;

            // cout_mutex.lock();
            // std::cout << "[Thread " << tid << "] runs out of free men"
            //           << std::endl;
            // cout_mutex.unlock();
          }
          p_rank = m_rank;
        } else {
          p_rank = expected;
        }
      }
    }
  }

  // ****************** GetMatch Parallel CPU***************************
  // *******************************************************************
  auto GS::GetMatchVectorParallelCPU() const -> std::vector<int> {
    std::vector<int> match_vec(n_);

    auto start_post = getNanoSecond();
    for (int i = 0; i < n_; ++i) {
      husband_rank_[i] = atomic_partner_rank_[i].load();
    }

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = husband_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }

    auto end_post = getNanoSecond();
    float post_time = (end_post - start_post) * 1.0 / 1e6;
    std::cout << "Postprocess takes " << post_time << " milliseconds"
        << std::endl;
    return match_vec;
  }
}
