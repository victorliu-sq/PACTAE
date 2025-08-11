#include "smp/smp_engine_mw_1_f7.h"

#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineMwF7::SmpEngineMwF7(const SmpObj &smp, size_t n)
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


  void SmpEngineMwF7::InitProc() {
    // Init Rank Matrix
    const int n = this->n_;
    // 1. Init Rank Matrix
    auto pref_list_w_dview = this->dev_pref_lists_w_.DeviceView();
    auto rank_mtx_w_dview = this->dev_rank_mtx_w_.DeviceView();

    this->ExecuteNTasklet(SIZE_MUL(n, n), [=] __device__(size_t tid) mutable {
      int m_idx, w_idx, m_rank;
      w_idx = tid / n;
      m_rank = tid % n;
      m_idx = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, m_rank)];
      rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)] = m_rank;
    });

    this->dev_rank_mtx_w_.GetFromDevice(this->host_rank_mtx_w_);

    // Init Other Metadata
    for (int i = 0; i < n_; ++i) {
      partner_rank_[i] = n_;
      next_proposed_w_[i] = 0;
    }
  }

  void SmpEngineMwF7::CoreProc() {
    for (int m = 0; m < n_; m++) {
      this->MwProcedure(m);
    }
  }

  void SmpEngineMwF7::MwProcedure(int m) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = this->next_proposed_w_[m_idx];
    bool is_matched = false;

    // auto flatten_pref_lists_m_ptr = RawPtr(smp_.flatten_pref_lists_m_vec);
    // auto flatten_pref_lists_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

    while (!is_matched) {
      w_idx = smp_.flatten_pref_lists_m_vec[m_idx * n_ + w_rank];
      // w_idx = smp_.flatten_pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
      // w_idx = flatten_pref_lists_m_ptr[IDX_MUL_ADD(m_idx, n_, w_rank)];

      // m_rank = this->rank_mtx_w_[w_idx * n_ + m_idx];
      m_rank = this->host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

      // std::cout << "Man " << m_idx << " proposed to " << w_rank << " th Woman "
      //           << w_idx;
      w_rank += 1;
      p_rank = this->partner_rank_[w_idx];
      if (p_rank == n_) {
        this->next_proposed_w_[m_idx] = w_rank;
        this->partner_rank_[w_idx] = m_rank;
        is_matched = true;
        // printf("and woman %d is unpaired, so STOP\n", w_idx);
      } else if (p_rank > m_rank) {
        this->next_proposed_w_[m_idx] = w_rank;
        this->partner_rank_[w_idx] = m_rank;

        m_idx = smp_.flatten_pref_lists_w_vec[w_idx * n_ + p_rank];
        // m_idx = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
        // m_idx = flatten_pref_lists_w_ptr[IDX_MUL_ADD(w_idx, n_, p_rank)];
        w_rank = this->next_proposed_w_[m_idx];
        // printf("and gets ACCEPTED. Since woman %d is paired with man %d, so
        // "SWITCH\n", w_idx, m_idx);
      } else {
        // printf("and gets REJECTED, so move to next woman\n");
      }
    }
  }


  auto SmpEngineMwF7::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
