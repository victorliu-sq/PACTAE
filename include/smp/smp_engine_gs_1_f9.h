#ifndef SMP_ENGINE_GS_1_F9_H
#define SMP_ENGINE_GS_1_F9_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineGsF9 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineGsF9(const SmpObj &smp, const size_t n);

    ~SmpEngineGsF9() {
    }

    auto GetEngineName() const -> String override {
      return "Gs-Seq-CPU:InitSeqCPU+ExecSeqCPU";
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
    // const int num_threads_;

    Vector<int> rank_mtx_w_;

    // queue of free men
    Queue<int> free_men_queue_;

    Vector<int> next_proposed_w_;

    Vector<int> partner_rank_;
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_GS_1_F9_H
