#ifndef SMP_ENGINE_MW_4_F8_TLEFT_H
#define SMP_ENGINE_MW_4_F8_TLEFT_H

#include "smp.h"
#include "smp_engine_abs.h"
#include "utils/dev_array.h"
#include "utils/stream.h"
#include "utils/launcher.h"

namespace bamboosmp {
  class SmpEngineMw4F8TLeft : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMw4F8TLeft(const SmpObj &smp, size_t n);

    ~SmpEngineMw4F8TLeft() = default;

    auto GetEngineName() const -> String override {
      return "MW-Par-GPU-MIN:Figure8-TLeft";
    }

    auto IsPerfect() const -> bool override {
      return false;
    }

    void InitProc() override;

    void CoreProc() override;

    auto PostProc() -> Matching override;

  private:
    const SmpObj &smp_;
    const size_t n_;

    CudaStream cuda_stream_{};

    // SMPObj in the device
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;

    // Metadata in the device
    DArray<int> dev_next_proposed_w_;
    DArray<int> dev_partner_rank_;

    // Metadata in the host for postprocessing
    HArray<int> host_partner_rank_;

    // ------------ cuda utility methods -----------------------------
    template<typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEach(this->cuda_stream_, n, f, arg...);
    }
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_MW_4_F8_TLEFT_H
