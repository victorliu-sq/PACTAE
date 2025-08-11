#pragma once

#include "smp.h"
#include <atomic>
#include <vector>
#include <utils/stream.h>

#include "atomic_tflag.h"
#include "smp/smp_engine_abstract_old.h"
#include "utils/timer.h"
#include "utils/types.h"
#include "kernel.h"

namespace bamboosmp {
  class SmpEngineLaOld : public OldAbsSmpEngine {
  public:
    // Constructor
    SmpEngineLaOld(const SmpObj &smp, int size);

    ~SmpEngineLaOld() = default;

    auto GetStableMarriage() const -> std::vector<int>;

    void PrintMatching() const;

  protected:
    inline auto IsPerfect() const -> bool override {
      return is_perfect_;
    }

    void InitProc() override;

    void CoreProc() override;

    void PostProc() override;

  private:
    const SmpObj &smp_;
    int n_;
    CudaStream default_stream_{};
    CudaStream monitor_stream_{};
    const int *pref_lists_w_view_;

    bool is_perfect_;
    HVector<int> stable_matching_;

    DVector<int> device_pref_lists_m_;
    DVector<int> device_pref_lists_w_;
    DVector<int> device_rank_mtx_w_;
    HVector<int> host_rank_mtx_w_;

    DVector<PRNode> device_prnodes_m_;
    PRNode *host_prnodes_m_ptr_;

    // for sequential algorithm
    void LAProcedure(int m);

    void LAParallelProcedure(int m);

    // for parallel algorithm on GPU
    HVector<int> host_partner_rank_;
    HVector<int> temp_host_partner_rank_;
    DVector<int> device_partner_rank_;

    DVector<int> device_free_man_idx_;
    DVector<int> device_num_unproposed_;

    HVector<int> host_next_proposed_w_;
    DVector<int> device_next_proposed_w_;

    std::atomic<int> *atomic_partner_rank_;
  };
} // namespace bamboosmp
