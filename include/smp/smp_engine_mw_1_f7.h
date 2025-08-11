#ifndef SMP_ENGINE_MW_1_F7_H
#define SMP_ENGINE_MW_1_F7_H

#include "smp.h"
#include "smp_engine_abs.h"
#include "utils/dev_array.h"
#include "utils/stream.h"
#include "utils/launcher.h"

namespace bamboosmp {
  class SmpEngineMwF7 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMwF7(const SmpObj &smp, size_t n);

    ~SmpEngineMwF7() {
    }

    auto GetEngineName() const -> String override {
      return "Mw-Seq-CPU:Figure7";
    }

    auto IsPerfect() const -> bool override {
      return false;
    }

    void InitProc() override;

    void CoreProc() override;

    auto PostProc() -> Matching override;

  private:
    void MwProcedure(int m);

    const SmpObj &smp_;
    const size_t n_;
    // const int num_threads_;

    Vector<int> host_rank_mtx_w_;

    Vector<int> next_proposed_w_;

    Vector<int> partner_rank_;

    // void MWProcedure(int m);
    // ----------- Figure 7 Config ---------------
    // SMPObj in the device
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    // DArray<PRNode> dev_prmtx_;

    CudaStream cuda_stream_{};

    // ------------ cuda utility methods -----------------------------
    template<typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEach(this->cuda_stream_, n, f, arg...);
    }
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_MW_1_F7_H
