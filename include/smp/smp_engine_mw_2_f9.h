#ifndef SMP_ENGINE_MW_2_F9_H
#define SMP_ENGINE_MW_2_F9_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineMw2F9 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMw2F9(const SmpObj &smp, size_t n);

    ~SmpEngineMw2F9();

    auto GetEngineName() const -> String override {
      return "Mw-Par-Cpu:InitSeqCPU+ExecParCPU";
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

    Vector<int> rank_mtx_w_;

    Vector<int> next_proposed_w_;

    std::atomic<int> *atomic_partner_rank_;
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_MW_2_F9_H
