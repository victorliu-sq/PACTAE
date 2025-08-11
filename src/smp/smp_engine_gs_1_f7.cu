#include "smp/smp_engine_gs_1_f7.h"
#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineGsF7::SmpEngineGsF7(const SmpObj &smp, const size_t n)
    : smp_(smp),
      n_(n),
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      host_rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_),
      partner_rank_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
  }

  void SmpEngineGsF7::InitProc() {
    // for (size_t w_idx = 0; w_idx < n_; ++w_idx) {
    //   for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
    //     // size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
    //     size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
    //     rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
    //   }
    // }
    const int n = this->n_;
    // 1. Init Rank Matrix
    auto pref_list_w_dview = this->dev_pref_lists_w_.DeviceView();
    auto rank_mtx_w_dview = this->dev_rank_mtx_w_.DeviceView();

    this->ExecuteNTasklet(SIZE_MUL(n, n), [=] __device__(size_t tid) mutable {
      int m_idx, w_idx, m_rank;
      w_idx = tid / n;
      m_rank = tid % n;
      // m_idx = pref_lists[w_idx * n + m_rank];
      m_idx = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, m_rank)];
      // rank_mtx[w_idx * n + m_idx] = m_rank;
      rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)] = m_rank;
    });

    this->dev_rank_mtx_w_.GetFromDevice(this->host_rank_mtx_w_);

    // Init Other Metadata
    for (int i = 0; i < n_; ++i) {
      free_men_queue_.push(i);
      partner_rank_[i] = n_;
      next_proposed_w_[i] = 0;
    }
  }

  void SmpEngineGsF7::CoreProc() {
    int m_idx, m_rank, w_idx, w_rank, p_rank;

    m_idx = free_men_queue_.front();
    free_men_queue_.pop();
    w_rank = 0;
    bool done = false;
    int iteration = 0;

    while (!done) {
      iteration += 1;
      w_idx = smp_.flatten_pref_lists_m_vec[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // w_idx = flatten_pref_lists_m_ptr[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // w_idx = smp_.flatten_pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];

      m_rank = host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

      w_rank += 1;

      p_rank = partner_rank_[w_idx];

      if (m_rank < p_rank) {
        partner_rank_[w_idx] = m_rank;
        if (p_rank != n_) {
          int new_free_man = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, p_rank)];
          // int new_free_man = flatten_pref_lists_w_ptr[IDX_MUL_ADD(w_idx, n_, p_rank)];
          // int new_free_man = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
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

  auto SmpEngineGsF7::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      // int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
