#ifndef SMP_ENGINE_MW_1_F9_H
#define SMP_ENGINE_MW_1_F9_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineMwF9 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMwF9(const SmpObj &smp, size_t n);

    ~SmpEngineMwF9() {
    }

    auto GetEngineName() const -> String override {
      return "Mw-Seq-CPU:InitSeqCPU+ExecSeqCPU";
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

    Vector<int> rank_mtx_w_;

    Vector<int> next_proposed_w_;

    Vector<int> partner_rank_;

  };
} // namespace bamboosmp

#endif //SMP_ENGINE_MW_1_F9_H
