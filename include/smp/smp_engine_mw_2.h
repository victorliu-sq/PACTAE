#ifndef SMP_ENGINE_MW_PAR_H
#define SMP_ENGINE_MW_PAR_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineMw2 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMw2(const SmpObj &smp, size_t n);

    ~SmpEngineMw2();

    auto GetEngineName() const -> String override {
      return "Mw-Par-Cpu";
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

#endif //SMP_ENGINE_MW_PAR_H
