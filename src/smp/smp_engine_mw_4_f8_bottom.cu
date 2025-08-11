#include "smp/smp_engine_mw_4_f8_bottom.h"
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

#include "smp/kernels.h"

namespace bamboosmp {
  SmpEngineMw4F8Bottom::SmpEngineMw4F8Bottom(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      dev_next_proposed_w_(n_),
      dev_partner_rank_(n_),
      host_partner_rank_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
  }

  void SmpEngineMw4F8Bottom::InitProc() {
    // Initialize Rank Matrix
    const int n = this->n_;
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

    // Initialialze Metadata
    this->dev_partner_rank_.Fill(n);
    // for (int i = 0; i < n; i++) {
    //   std::cout << i << " : " << this->dev_partner_rank_[i] << std::endl;
    // }
    //
    this->dev_next_proposed_w_.Fill(0);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void SmpEngineMw4F8Bottom::CoreProc() {
    // const int n = this->n_;
    const int n = this->n_;

    const int *pref_list_m_dview = RawPtr(this->dev_pref_lists_m_);
    const int *pref_list_w_dview = RawPtr(this->dev_pref_lists_w_);
    const int *rank_mtx_w_dview = RawPtr(this->dev_rank_mtx_w_);
    int *next_proposed_w_dview = RawPtr(this->dev_next_proposed_w_);
    int *partner_rank_dview = RawPtr(this->dev_partner_rank_);

    const int block_size = 1024;
    const int grid_size = (n + block_size - 1) / block_size;

    Mw4F8BottomKernel<<<grid_size, block_size, 0, cuda_stream_.cuda_stream()>>>(
      n,
      pref_list_m_dview,
      pref_list_w_dview,
      rank_mtx_w_dview,
      next_proposed_w_dview,
      partner_rank_dview
    );

    cuda_stream_.Sync();
  }

  auto SmpEngineMw4F8Bottom::PostProc() -> Matching {
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
