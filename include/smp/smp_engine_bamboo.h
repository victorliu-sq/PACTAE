#ifndef SMP_ENGINE_BAMBOO_H
#define SMP_ENGINE_BAMBOO_H

#include <set>

#include "smp.h"
#include "smp_engine_abs.h"
#include "utils/dev_array.h"
#include "utils/stream.h"
#include "utils/launcher.h"

namespace bamboosmp {
  class SmpEngineBamboo : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineBamboo(const SmpObj &smp, size_t n);

    ~SmpEngineBamboo() {
    }

    auto GetEngineName() const -> String override {
      return "BambooSMP";
    }

    auto IsPerfect() const -> bool override {
      std::set<int> topChoicesSet;
      for (int m = 0; m < n_; ++m) {
        int topChoice = smp_.flatten_pref_lists_m[IDX_MUL(m, n_)];
        topChoicesSet.insert(topChoice);
        // stable_matching_[m] = topChoice;
      }
      return topChoicesSet.size() == n_;
    }

    void InitProc() override;

    void CoreProc() override;

    auto PostProc() -> Matching override;

    // Lambda must be executed in a public method
    void DoWorkOnGpu();

  private:
    void DoWorkOnCpu();

    void MonitorProceture();

    inline void AsyncD2HPRMatrix() {
      CUDA_CHECK(cudaMemcpyAsync(RawPtr(this->host_prmtx_),
        RawPtr(this->dev_prmtx_),
        SIZE_MUL(n_, n_) * sizeof(PRNode),
        cudaMemcpyDeviceToHost,
        monitor_stream_.cuda_stream()));

      monitor_stream_.Sync();
    }

    inline void AsyncD2HPartnerRank() {
      CUDA_CHECK(cudaMemcpyAsync(RawPtr(this->host_partner_rank_),
        RawPtr(this->dev_partner_rank_),
        n_ * sizeof(int),
        cudaMemcpyDeviceToHost,
        this->monitor_stream_.cuda_stream()));
      this->monitor_stream_.Sync();
    }

    inline void AsyncD2HNext() {
      CUDA_CHECK(cudaMemcpyAsync(RawPtr(this->host_next_proposed_w_),
        RawPtr(this->dev_next_proposed_w_),
        n_ * sizeof(int),
        cudaMemcpyDeviceToHost,
        this->monitor_stream_.cuda_stream()));
      this->monitor_stream_.Sync();
    }

    void LAProcedure(int m);

    // ------------ cuda utility methods -----------------------------
    template<typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEach(this->default_stream_, n, f, arg...);
    }

    // ------------ Data Members -----------------------------
    const SmpObj &smp_;
    const size_t n_;
    const int num_threads_;

    CudaStream default_stream_{};
    CudaStream monitor_stream_{};

    // SMPObj in the device
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    DArray<PRNode> dev_prmtx_;

    // Metadata in the device
    DArray<int> dev_next_proposed_w_;
    DArray<int> dev_partner_rank_;

    // SMPObject in the host
    // HArray<PRNode> host_prmtx_;
    Vector<PRNode> host_prmtx_;

    // Metadata in the host for postprocessing
    Vector<int> host_next_proposed_w_;
    Vector<int> host_partner_rank_;

    // For Monitor
    int unmatched_num_;
    int unmatched_id_;
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_BAMBOO_H
