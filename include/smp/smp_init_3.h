#ifndef SMP_INIT_3_H
#define SMP_INIT_3_H

#include "smp.h"
#include "smp/smp_init_abs.h"
#include "utils/dev_array.h"
#include "utils/stream.h"
#include "utils/launcher.h"

namespace bamboosmp {
  class SmpInitEngine3 : public AbsSmpInitEngine {
  public:
    SmpInitEngine3(const SmpObjInit &smp, size_t n, bool init_pr_mtx)
      : AbsSmpInitEngine(init_pr_mtx),
        smp_(smp),
        n_(n),
        num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
        // Device Smp Obj
        dev_pref_lists_m_(SIZE_MUL(n_, n_)),
        dev_pref_lists_w_(SIZE_MUL(n_, n_)),
        dev_rank_mtx_w_(SIZE_MUL(n_, n_)),
        dev_prmtx_(SIZE_MUL(n_, n_)),
        host_rank_mtx_w_(SIZE_MUL(n_, n_)),
        host_prmtx_(SIZE_MUL(n_, n_)) {
      // Copy Smp Obj from Host to Device
      dev_pref_lists_m_.SetToDevice(smp_.flatten_pref_lists_m_vec);
      dev_pref_lists_w_.SetToDevice(smp_.flatten_pref_lists_w_vec);
    }

    auto GetEngineName() const -> String override {
      return "Smp-Init-Gpu";
    }

    void PrintProfilingInfo() override {
      this->total_init_time_ = this->init_rank_mtx_time_;
      if (this->init_pr_mtx_) {
        this->total_init_time_ += init_pr_mtx_time_;
      }

      ms_t kernel_time = this->total_init_time_ - this->transfer_time_;

      // Title
      OutStringStream oss_title;
      oss_title << this->GetEngineName();
      if (this->init_pr_mtx_) {
        oss_title << " Initialize Both RankMatrix and PRMatrix ";
      } else {
        oss_title << " Initialize RankMatrix Only ";
      }
      oss_title << "Profiling Info: ";
      String title = oss_title.str();

      std::cout << title << std::endl;
      std::cout << this->GetEngineName() << " Kernel Time: " << kernel_time << " ms" << std::endl;
      std::cout << this->GetEngineName() << " Host-Device Transfer Time: " << this->transfer_time_ << " ms" <<
          std::endl;
      std::cout << this->GetEngineName() << " Total Init Time: " << this->total_init_time_ << " ms" << std::endl;
      std::cout << "==================================================================" << std::endl;

      LOG(INFO) << title;
      LOG(INFO) << this->GetEngineName() << " Kernel Time: " << kernel_time << " ms";
      LOG(INFO) << this->GetEngineName() << " Host-Device Transfer Time: " << this->transfer_time_ << " ms";
      LOG(INFO) << this->GetEngineName() << " Total Init Time: " << this->total_init_time_ << " ms";
      LOG(INFO) << "==================================================================";
    }

    // Set to public due to requirement of CUDA lambda
    void InitRankMatrix() override {
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

      // D2H
      if (!this->init_pr_mtx_) {
        StopWatch d2h_sw;
        d2h_sw.Start();

        this->dev_rank_mtx_w_.GetFromDeviceDirectly(this->host_rank_mtx_w_);
        CUDA_CHECK(cudaDeviceSynchronize());

        d2h_sw.Stop();
        this->transfer_time_ = d2h_sw.GetEclapsedMs();
      }
    }

    void InitPRMatrix() override {
      const int n = this->n_;
      auto pref_list_m_dview = this->dev_pref_lists_m_.DeviceView();
      auto rank_mtx_w_dview = this->dev_rank_mtx_w_.DeviceView();
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

      // D2H
      if (this->init_pr_mtx_) {
        StopWatch d2h_sw;
        d2h_sw.Start();

        this->dev_prmtx_.GetFromDeviceDirectly(this->host_prmtx_);

        d2h_sw.Stop();
        this->transfer_time_ = d2h_sw.GetEclapsedMs();
      }
    }

  private:
    // Profiling Information
    ms_t transfer_time_{0};

    // Smp Information
    const SmpObjInit &smp_;
    const size_t n_;
    const int num_threads_;

    CudaStream cuda_stream_{};

    // SMPObj in the device
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    DArray<PRNode> dev_prmtx_;

    // SMPObject in the host
    HVector<int> host_rank_mtx_w_;
    HVector<PRNode> host_prmtx_;

    // ------------ cuda utility methods -----------------------------
    template<typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEachMax(this->cuda_stream_, n, f, arg...);
    }
  };
}

#endif //SMP_INIT_3_H
