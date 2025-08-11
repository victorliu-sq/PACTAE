#include "smp/smp_engine_la_2_f7.h"

#include "smp/kernels.h"

namespace bamboosmp {
  SmpEngineLa2F7::SmpEngineLa2F7(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      host_rank_mtx_w_(n_ * n_),
      dev_prmtx_(SIZE_MUL(n_, n_)),
      dev_next_proposed_w_(n_),
      dev_partner_rank_(n_),
      host_partner_rank_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
  }

  void SmpEngineLa2F7::InitProc() {
    // Initialialze Dev Metadata
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

    this->dev_rank_mtx_w_.GetFromDevice(this->host_rank_mtx_w_);

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

      // node.idx_ = w_idx;
      // node.rank_ = m_rank;
      // prnodes_m[m_idx * n + w_rank] = {w_idx, m_rank};
      prmtx_dview[prnode_id] = {w_idx, m_rank};
    });
  }

  void SmpEngineLa2F7::CoreProc() {
    const int n = this->n_;
    // auto prmtx_dview = this->dev_prmtx_.DeviceView();
    //
    // auto pref_list_w_dview = this->dev_pref_lists_w_.DeviceView();
    // auto next_proposed_w_dview = this->dev_next_proposed_w_.DeviceView();
    // auto partner_rank_dview = this->dev_partner_rank_.DeviceView();
    //
    // this->ExecuteNTasklet(this->n_, [=] __device__(size_t tid) mutable {
    //   int mi, mi_rank, w_idx, w_rank, mj, mj_rank, mj_rank2;
    //   mi = tid;
    //   w_rank = 0;
    //   PRNode node;
    //   bool is_married = false;
    //   while (!is_married) {
    //     // w_idx = pref_list_m_dview[mi * n + w_rank];
    //     // printf("man %d proposes to woman %d\n", mi, w_idx);
    //     // mi_rank = rank_mtx_w_dview[w_idx * n + mi];
    //     node = prmtx_dview[IDX_MUL_ADD(mi, n, w_rank)];
    //     w_idx = node.idx_;
    //     mi_rank = node.rank_;
    //
    //     w_rank += 1;
    //     mj_rank = partner_rank_dview[w_idx];
    //
    //     // Figure 7
    //     // if (mj_rank < mi_rank) {
    //     //   continue;
    //     // }
    //
    //     while (mj_rank > mi_rank) {
    //       mj_rank2 = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
    //       if (mj_rank2 == mj_rank) {
    //         next_proposed_w_dview[mi] = w_rank;
    //         if (mj_rank == n) {
    //           is_married = true;
    //         } else if (mj_rank > mi_rank) {
    //           mi = pref_list_w_dview[w_idx * n + mj_rank];
    //           w_rank = next_proposed_w_dview[mi];
    //         }
    //         break;
    //       } else {
    //         mj_rank = mj_rank2;
    //       }
    //     }
    //   }
    // });

    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;

    La2F7CoreKernel<<<grid_size, block_size, 0, cuda_stream_.cuda_stream()>>>(
      n,
      RawPtr(dev_prmtx_),
      RawPtr(dev_partner_rank_),
      RawPtr(dev_next_proposed_w_),
      RawPtr(dev_pref_lists_w_)
    );

    cuda_stream_.Sync();
  }

  auto SmpEngineLa2F7::PostProc() -> Matching {
    this->dev_partner_rank_.GetFromDevice(this->host_partner_rank_);

    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = this->host_partner_rank_[w_id];
      int m_id = this->smp_.flatten_pref_lists_w[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }

    return match_vec;
  }
}
