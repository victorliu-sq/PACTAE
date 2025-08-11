#include "smp/smp_engine_bamboo.h"

#include "smp/kernels.h"

namespace bamboosmp {
  SmpEngineBamboo::SmpEngineBamboo(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
      // Host-SmpObj
      host_prmtx_(SIZE_MUL(n_, n_)),
      // Dev-SmpObj
      dev_pref_lists_m_(SIZE_MUL(n_, n_)),
      dev_pref_lists_w_(SIZE_MUL(n_, n_)),
      dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
      dev_prmtx_(SIZE_MUL(n_, n_)),
      // Dev-MetaData
      dev_next_proposed_w_(n_),
      dev_partner_rank_(n_),
      // Host-MetaData
      host_next_proposed_w_(n_),
      host_partner_rank_(n_),
      // Monitor MetaData
      unmatched_id_(n_),
      unmatched_num_(n_) {
    // Copy Smp Obj from Host to Device
    dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
    dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec); {
    }
  }

  void SmpEngineBamboo::InitProc() {
    const int n = this->n_;

    // ------------ Initialialze Dev Metadata ---------------------------------------
    this->dev_partner_rank_.Fill(n);
    this->dev_next_proposed_w_.Fill(0);

    // ------------ Initialialze Host Metadata ---------------------------------------
    for (int i = 0; i < n_; ++i) {
      this->host_partner_rank_[i] = n;
      this->host_next_proposed_w_[i] = 0;
    }

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
    StopWatch sw;
    sw.Start();
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
    sw.Stop();
    std::cout << "[Init PRMatrix]: store both values takes " << sw.GetEclapsedMs() << " ms" << std::endl;
  }

  void SmpEngineBamboo::CoreProc() {
    std::thread thread_gpu(&SmpEngineBamboo::DoWorkOnGpu, this);
    this->DoWorkOnCpu();
    thread_gpu.detach();
  }

  void SmpEngineBamboo::DoWorkOnGpu() {
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
    //     // node = prnodes_m[mi * n + w_rank];
    //     node = prmtx_dview[IDX_MUL_ADD(mi, n, w_rank)];
    //     w_idx = node.idx_;
    //     mi_rank = node.rank_;
    //     w_rank += 1;
    //
    //     // if (partner_rank[w_idx] < mi_rank) {
    //     //   continue;
    //     // }
    //
    //     mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    //     if (mj_rank > mi_rank) {
    //       next_proposed_w_dview[mi] = w_rank;
    //       if (mj_rank == n) {
    //         paired = true;
    //       } else {
    //         // mi = pref_lists_w[w_idx * n + mj_rank];
    //         mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
    //         w_rank = next_proposed_w_dview[mi];
    //       }
    //     }
    //   }
    // });

    int block_size = 64;
    int grid_size = (n + block_size - 1) / block_size;

    La3F9CoreKernel<<<grid_size, block_size, 0, default_stream_.cuda_stream()>>>(
      n,
      RawPtr(dev_prmtx_),               // PRNode* [n*n]
      RawPtr(dev_partner_rank_),        // int*     [n]
      RawPtr(dev_next_proposed_w_),    // int*     [n]
      RawPtr(dev_pref_lists_w_)         // int*     [n*n]
    );

    default_stream_.Sync();
  }

  void SmpEngineBamboo::DoWorkOnCpu() {
    this->MonitorProceture(); // exit when unmatched_num is either 0 or 1.
    if (this->unmatched_num_ == 1) {
      this->AsyncD2HPRMatrix();
      // CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_view_,
      //   RawPtr(device_prnodes_m_),
      //   SIZE_MUL(n_, n_) * sizeof(PRNode),
      //   cudaMemcpyDeviceToHost,
      //   monitor_stream_.cuda_stream()));
      //
      // monitor_stream_.Sync();
      // this->AsyncD2HNext();

      // LOG(INFO) << "CPU starts LAProcedure.";
      // auto start_la = getNanoSecond();
      // std::cout << "Unmmatched man is " << unmatched_id_;
      this->LAProcedure(this->unmatched_id_);
      // auto stop_la = getNanoSecond();
      // std::cout << "[BambooSMP::LAProcedure] takes time: " << (stop_la - start_la) / 1e6 << " ms." << std::endl;

      LOG(INFO) << "CheckKernel (CPU) won the contention.";
    } else {
      LOG(INFO) << "CheckKernel (GPU) won the contention.";
    }
  }

  inline void SmpEngineBamboo::LAProcedure(int m) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node{};
    bool is_matched = false;
    int iterations = 0;

    while (!is_matched) {
      iterations += 1;
      // temp_node = host_prnodes_m_view_[m_idx * n_ + w_rank];
      // temp_node = this->host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];
      temp_node = host_prmtx_[IDX_MUL_ADD(m_idx, n_, w_rank)];

      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      p_rank = host_partner_rank_[w_idx];
      if (p_rank == n_) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;
        is_matched = true;
      } else if (p_rank > m_rank) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;

        // m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
        m_idx = this->smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
        w_rank = host_next_proposed_w_[m_idx];
      } else {
        w_rank++;
      }
    }
  }

  void SmpEngineBamboo::MonitorProceture() {
    int it = 0;
    // const int total = n_ * (n_ - 1) / 2;
    const size_t total = SIZE_MUL(n_, (n_ - 1)) / 2;
    bool encountered_1_once = false;

    do {
      // timer_start(true);
      // timer_next("Copy Rank D2H");
      SLEEP_MILLISECONDS(10);

      // CUDA_CHECK(cudaMemcpyAsync(RawPtr(this->host_partner_rank_),
      //   RawPtr(this->dev_partner_rank_),
      //   static_cast<size_t>(n_) * sizeof(int),
      //   cudaMemcpyDeviceToHost,
      //   this->monitor_stream_.cuda_stream()));
      // this->monitor_stream_.Sync();
      this->AsyncD2HPartnerRank();

      this->unmatched_id_ = total;
      this->unmatched_num_ = 0;

      // timer_next("Iterate N Ranks");
      // for (int w = 0; w < n_; w++) {
      //   if (host_partner_rank_[w] == n_) {
      //     this->unmatched_num_++;
      //   } else {
      //     int m_rank = host_partner_rank_[w];
      //     // this->unmatched_id_ -= pref_lists_w_view_[w * n_ + m_rank];
      //     // this->unmatched_id_ -= pref_lists_w_view_[IDX_MUL_ADD(w, n_, m_rank)];
      //     this->unmatched_id_ -= this->smp_.flatten_pref_lists_w[IDX_MUL_ADD(w, n_, m_rank)];
      //   }
      // }
      for (int w = 0; w < n_; w++) {
        if (host_partner_rank_[w] == n_) {
          this->unmatched_num_++;
        } else {
          int m_rank = host_partner_rank_[w];
          this->unmatched_id_ -= this->smp_.flatten_pref_lists_w[IDX_MUL_ADD(w, n_, m_rank)];
        }
      }

      if (this->unmatched_num_ == 0) {
        break; // Exit immediately if 0 unmatched
      } else if (this->unmatched_num_ == 1 && !encountered_1_once) {
        encountered_1_once = true;
        continue; // Go into second observation loop
      } else if (this->unmatched_num_ <= 1 && encountered_1_once) {
        break; // Second observation loop: exit no matter if 0 or 1
      }

      // timer_end();
      // LOG(INFO) << "Iteration " << it << ": # of unmatched men = " << this->unmatched_num_;
      it++;
    } while (this->unmatched_num_ > 1);
  }

  auto SmpEngineBamboo::PostProc() -> Matching {
    Matching match_vec(n_);
    if (this->IsPerfect()) {
      // Perfect case
      for (int m = 0; m < n_; ++m) {
        // int topChoice = smp_.flatten_pref_lists_m_[m * n_];
        int topChoice = smp_.flatten_pref_lists_m[IDX_MUL(m, n_)];
        match_vec[m] = topChoice;
      }
    } else {
      for (int w_id = 0; w_id < match_vec.size(); w_id++) {
        int m_rank = host_partner_rank_[w_id];
        int m_id = smp_.flatten_pref_lists_w[w_id * n_ + m_rank];
        match_vec[m_id] = w_id;
      }
    }
    return match_vec;
  }
}
