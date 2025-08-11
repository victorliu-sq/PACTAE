#ifndef SMP_ENGINE_MW_SEQ_H
#define SMP_ENGINE_MW_SEQ_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineMw : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMw(const SmpObj &smp, size_t n);

    ~SmpEngineMw() {
    }

    auto GetEngineName() const -> String override {
      return "Mw-Seq-CPU";
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

    Vector<int> partner_rank_;

    // void MWProcedure(int m);
  };
} // namespace bamboosmp


#endif //SMP_ENGINE_MW_SEQ_H
