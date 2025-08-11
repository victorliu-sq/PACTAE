#ifndef SMP_ENGINE_LA_H
#define SMP_ENGINE_LA_H

#include "smp.h"
#include "smp_engine_abs.h"
#include "utils/dev_array.h"
#include "utils/stream.h"
#include "utils/launcher.h"

namespace bamboosmp {
  class SmpEngineLaF7 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineLaF7(const SmpObj &smp, size_t n);

    ~SmpEngineLaF7() {
    }

    auto GetEngineName() const -> String override {
      return "LA-Seq-Cpu:Figure7";
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
    const int num_threads_;

    CudaStream cuda_stream_{};

    // SMPObj in the device
    DArray<int> dev_pref_lists_m_;
    DArray<int> dev_pref_lists_w_;
    DArray<int> dev_rank_mtx_w_;
    DArray<PRNode> dev_prmtx_;

    // Metadata in the device
    // DArray<int> dev_next_proposed_w_;
    // DArray<int> dev_partner_rank_;

    // SMPObject in the host
    // HArray<PRNode> host_prmtx_;
    Vector<PRNode> host_prmtx_;

    // Metadata in the host for postprocessing
    Vector<int> next_proposed_w_;
    // std::atomic<int> *atomic_partner_rank_;
    Vector<int> partner_rank_;

    // ------------ cuda utility methods -----------------------------
    template<typename F, typename... Arg>
    void ExecuteNTasklet(size_t n, F f, Arg... arg) {
      LaunchKernelForEach(this->cuda_stream_, n, f, arg...);
    }

    inline void LaProcedure(int m);
  };
} // namespace bamboosmp


#endif //SMP_ENGINE_LA_H
