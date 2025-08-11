#include "smp/smp_engine_la_f7.h"

namespace bamboosmp {
  SmpEngineLaF7::SmpEngineLaF7(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      dev_prmtx_(SIZE_MUL(n_, n_)),
      host_prmtx_(SIZE_MUL(n_, n_)),
      next_proposed_w_(n_),
      partner_rank_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
  }

  void SmpEngineLaF7::InitProc() {
    for (int i = 0; i < n_; ++i) {
      next_proposed_w_[i] = 0;
      partner_rank_[i] = n_;
    }

    // ----------------- Initialize PRMatrix -----------------------------
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

    auto pref_list_m_dview = this->dev_pref_lists_m_.DeviceView();
    auto prmtx_dview = this->dev_prmtx_.DeviceView();

    // 2. Init PRMatrix
    this->ExecuteNTasklet(SIZE_MUL(n, n), [=] __device__(size_t tid) mutable {
      // PRNode node;
      int m_idx = tid / n;
      int w_rank = tid % n;

      // int w_idx = pref_lists_m[m_idx * n + w_rank];
      size_t prnode_id = IDX_MUL_ADD(m_idx, n, w_rank);
      int w_idx = pref_list_m_dview[prnode_id];
      // int m_rank = rank_mtx_w[w_idx * n + m_idx];
      int m_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)];

      // prnodes_m[m_idx * n + w_rank] = {w_idx, m_rank};
      prmtx_dview[prnode_id] = {w_idx, m_rank};
    });

    // read prmatrix from device
    this->dev_prmtx_.GetFromDevice(this->host_prmtx_);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void SmpEngineLaF7::CoreProc() {
    // LaProcedures
    for (int m = 0; m < n_; m++) {
      int w_idx, m_rank, m_idx, w_rank, p_rank;
      m_idx = m;
      w_rank = 0;
      PRNode temp_node{};
      bool is_matched = false;
      int iterations = 0;

      // auto host_prmtx_ptr = RawPtr(this->host_prmtx_);
      // auto flatten_pref_lists_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

      while (!is_matched) {
        iterations += 1;
        // temp_node = this->host_prmtx_[m_idx * n_ + w_rank];
        temp_node = this->host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];
        // temp_node = host_prmtx_ptr[IDX_MUL_ADD(m_idx, n_, w_rank)];
        w_idx = temp_node.idx_;
        m_rank = temp_node.rank_;
        p_rank = this->partner_rank_[w_idx];
        if (p_rank == n_) {
          this->next_proposed_w_[m_idx] = w_rank;
          this->partner_rank_[w_idx] = m_rank;
          is_matched = true;
        } else if (p_rank > m_rank) {
          this->next_proposed_w_[m_idx] = w_rank;
          this->partner_rank_[w_idx] = m_rank;

          // m_idx = this->smp_.flatten_pref_lists_w_[w_idx * n_ + p_rank];
          m_idx = this->smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
          // m_idx = flatten_pref_lists_w_ptr[IDX_MUL_ADD(w_idx, n_, p_rank)];
          w_rank = this->next_proposed_w_[m_idx];
        } else {
          w_rank++;
        }
      } // LaProcedure(m);
    }
  }

  inline void SmpEngineLaF7::LaProcedure(int m) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node{};
    bool is_matched = false;
    int iterations = 0;

    // auto host_prmtx_ptr = RawPtr(this->host_prmtx_);
    // auto flatten_pref_lists_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

    while (!is_matched) {
      iterations += 1;
      // temp_node = this->host_prmtx_[m_idx * n_ + w_rank];
      temp_node = this->host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];
      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      p_rank = this->partner_rank_[w_idx];
      if (p_rank == n_) {
        this->next_proposed_w_[m_idx] = w_rank;
        this->partner_rank_[w_idx] = m_rank;
        is_matched = true;
      } else if (p_rank > m_rank) {
        this->next_proposed_w_[m_idx] = w_rank;
        this->partner_rank_[w_idx] = m_rank;

        m_idx = this->smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, p_rank)];
        w_rank = this->next_proposed_w_[m_idx];
      } else {
        w_rank++;
      }
    }
  }

  auto SmpEngineLaF7::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
