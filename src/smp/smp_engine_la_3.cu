#include "smp/smp_engine_la_3.h"
#include "smp/kernels.h"

namespace bamboosmp {
  SmpEngineLa3::SmpEngineLa3(const SmpObj &smp, size_t n)
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

  void SmpEngineLa3::InitProc() {
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

      // int w_idx = pref_lists_m[m_idx * n + w_rank];
      int w_idx = pref_list_m_dview[IDX_MUL_ADD(m_idx, n, w_rank)];
      // int m_rank = rank_mtx_w[w_idx * n + m_idx];
      int m_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, m_idx)];

      // node.idx_ = w_idx;
      // node.rank_ = m_rank;
      // prnodes_m[m_idx * n + w_rank] = {w_idx, m_rank};
      prmtx_dview[IDX_MUL_ADD(m_idx, n, w_rank)] = {w_idx, m_rank};
    });
  }

  void SmpEngineLa3::CoreProc() {
    const int n = this->n_;
    // auto prmtx_dview = this->dev_prmtx_.DeviceView();
    //
    // auto pref_list_w_dview = this->dev_pref_lists_w_.DeviceView();
    // auto next_proposed_w_dview = this->dev_next_proposed_w_.DeviceView();
    // auto partner_rank_dview = this->dev_partner_rank_.DeviceView();
    //
    // this->ExecuteNTasklet(this->n_, [=] __device__(size_t tid) mutable {
    //   int mi, mi_rank, w_idx, w_rank, mj_rank;
    //   mi = tid;
    //   w_rank = 0;
    //   PRNode node;
    //   bool paired = false;
    //   while (!paired) {
    //     node = prmtx_dview[mi * n + w_rank];
    //     // node = prmtx_dview[IDX_MUL_ADD(mi, n, w_rank)];
    //     w_idx = node.idx_;
    //     mi_rank = node.rank_;
    //     w_rank += 1;
    //
    //     if (partner_rank_dview[w_idx] < mi_rank) {
    //       continue;
    //     }
    //
    //     mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    //     if (mj_rank > mi_rank) {
    //       next_proposed_w_dview[mi] = w_rank;
    //       if (mj_rank == n) {
    //         paired = true;
    //       } else {
    //         mi = pref_list_w_dview[w_idx * n + mj_rank];
    //         // mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
    //         w_rank = next_proposed_w_dview[mi];
    //       }
    //     }
    //   }
    // });

    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;

    La3MinF8BottomKernel<<<grid_size, block_size, 0, cuda_stream_.cuda_stream()>>>(
      n,
      RawPtr(dev_prmtx_), // PRNode* [n*n]
      RawPtr(dev_partner_rank_), // int*     [n]
      RawPtr(dev_next_proposed_w_), // int*     [n]
      RawPtr(dev_pref_lists_w_) // int*     [n*n]
    );

    cuda_stream_.Sync();
  }

  auto SmpEngineLa3::PostProc() -> Matching {
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
