#include "smp/smp.h"
#include <smp/gs.cuh>
#include <utils/timer.h>
#include <utils/utils.h>

namespace bamboosmp {
  auto GS::StartGS() -> float {
    PROFILE_SCOPE("Total time Gs-Seq");
    auto start_time_init = getNanoSecond();
    // InitGpu();
    // InitCpuSeq();
    InitCpuPar();
    auto stop_time_init = getNanoSecond();
    auto time_init = (stop_time_init - start_time_init) / 1e6;
    std::cout << "GS Init time is " << time_init << " ms." << std::endl;

    auto start_time_exec = getNanoSecond();
    int m_idx, m_rank, w_idx, w_rank, p_rank;
    const int *pref_lists_m = RawPtr(smp_.flatten_pref_lists_m_vec);
    const int *pref_lists_w = RawPtr(smp_.flatten_pref_lists_w_vec);

    m_idx = free_men_queue_.front();
    free_men_queue_.pop();
    w_rank = 0;
    bool done = false;
    int iteration = 0;
    while (!done) {
      iteration += 1;
      // w_idx = pref_lists_m[m_idx * n_ + w_rank];
      w_idx = pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // w_idx = smp_.flatten_pref_lists_m_[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // m_rank = host_rank_mtx_w_[w_idx * n_ + m_idx];
      m_rank = host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];
      // size_t safe_idx = IDX_MUL_ADD(w_idx, n_, m_idx);
      // m_rank = host_rank_mtx_w_[safe_idx];
      // std::cout << "Man " << m_idx << " proposed to " << w_rank << " th Woman "
      //     << w_idx;
      w_rank += 1;

      p_rank = husband_rank_[w_idx];
      // std::cout << "p_rank: " << p_rank << "; m_rank: " << m_rank << std::endl;

      if (m_rank < p_rank) {
        // std::cout << " And succeeds!" << std::endl;
        husband_rank_[w_idx] = m_rank;
        if (p_rank != n_) {
          // int new_free_man = pref_lists_w[w_idx * n_ + p_rank];
          int new_free_man = pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
          // int new_free_man = smp_.flatten_pref_lists_w_[IDX_MUL_ADD(w_idx, n_, p_rank)];

          free_men_queue_.push(new_free_man);
          // std::cout << "man " << new_free_man << " becomes free again"
          //     << std::endl;
        }
        next_proposed_w_[m_idx] = w_rank;
        if (!free_men_queue_.empty()) {
          m_idx = free_men_queue_.front();
          free_men_queue_.pop();
          w_rank = next_proposed_w_[m_idx];
        } else {
          done = true;
        }
      } else {
        // std::cout << " And fails" << std::endl;
      }
    }
    // std::cout << "number of iterations: " << iteration << std::endl;
    auto end_time_exec = getNanoSecond();
    auto time_exec = (end_time_exec - start_time_exec) / 1e6;
    std::cout << "Exec Phase-GS-Seq-CPU is: " << time_exec << std::endl;

    return time_init + time_exec;
  }

  auto GS::GetMatchVector() const -> std::vector<int> {
    std::vector<int> match_vec(n_);

    auto start_post = getNanoSecond();

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

  void GS::PrintMatches() {
    auto match_vec_row = GetMatchVector();
    std::cout << "*************** Matchings Start ***************" << std::endl;
    for (int i = 0; i < match_vec_row.size(); i++) {
      std::cout << "Matching " << i << " with " << match_vec_row[i] << std::endl;
    }
    std::cout << "*************** Matchings   End ***************" << std::endl;
  }
} // namespace bamboosmp
