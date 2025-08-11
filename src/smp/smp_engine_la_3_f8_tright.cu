#include "smp/smp_engine_la_3_f8_tright.h"
#include "smp/kernels.h"

namespace bamboosmp {
  SmpEngineLa3F8TRight::SmpEngineLa3F8TRight(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      dev_prmtx_(SIZE_MUL(n_, n_)),
      dev_next_proposed_w_(n_),
      dev_partner_rank_(n_),
      host_partner_rank_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
  }

  void SmpEngineLa3F8TRight::InitProc() {
    // Initialialze Metadata
    const int n = this->n_;
    this->dev_partner_rank_.Fill(n);
    this->dev_next_proposed_w_.Fill(0);

    // ----------------- Initialize PRMatrix -----------------------------
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

      int w_idx = pref_list_m_dview[IDX_MUL_ADD(m_idx, n, w_rank)];
      int m_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)];

      prmtx_dview[IDX_MUL_ADD(m_idx, n, w_rank)] = {w_idx, m_rank};
    });
  }

  void SmpEngineLa3F8TRight::CoreProc() {
    const int n = this->n_;

    int block_size = 512;
    int grid_size = (n + block_size - 1) / block_size;

    La3MinF8TopRightKernel<<<grid_size, block_size, 0, cuda_stream_.cuda_stream()>>>(
      n,
      RawPtr(dev_prmtx_), // PRNode* [n*n]
      RawPtr(dev_partner_rank_), // int*     [n]
      RawPtr(dev_next_proposed_w_), // int*     [n]
      RawPtr(dev_pref_lists_w_) // int*     [n*n]
    );

    cuda_stream_.Sync();
  }

  auto SmpEngineLa3F8TRight::PostProc() -> Matching {
    this->dev_partner_rank_.GetFromDevice(this->host_partner_rank_);

    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = this->host_partner_rank_[w_id];
      int m_id = this->smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }

    return match_vec;
  }
}
