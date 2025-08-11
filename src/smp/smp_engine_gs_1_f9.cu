#include "smp/smp_engine_gs_1_f9.h"
#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineGsF9::SmpEngineGsF9(const SmpObj &smp, const size_t n)
    : smp_(smp),
      n_(n),
      rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_),
      partner_rank_(n_) {
  }

  void SmpEngineGsF9::InitProc() {
    // Init Rank Matrix
    for (size_t w_idx = 0; w_idx < n_; ++w_idx) {
      for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
        size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
        rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
      }
    }


    // Init Other Metadata
    for (int i = 0; i < n_; ++i) {
      free_men_queue_.push(i);
      partner_rank_[i] = n_;
      next_proposed_w_[i] = 0;
    }
  }

  void SmpEngineGsF9::CoreProc() {
    int m_idx, m_rank, w_idx, w_rank, p_rank;

    m_idx = free_men_queue_.front();
    free_men_queue_.pop();
    w_rank = 0;
    bool done = false;
    int iteration = 0;

    while (!done) {
      iteration += 1;
      w_idx = smp_.flatten_pref_lists_m_vec[IDX_MUL_ADD(m_idx, n_, w_rank)];

      m_rank = rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

      w_rank += 1;

      p_rank = partner_rank_[w_idx];

      if (m_rank < p_rank) {
        partner_rank_[w_idx] = m_rank;
        if (p_rank != n_) {
          int new_free_man = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, p_rank)];
          free_men_queue_.push(new_free_man);
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
  }

  auto SmpEngineGsF9::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
