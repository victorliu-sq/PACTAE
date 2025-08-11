#ifndef SMP_ENGINE_GS_H
#define SMP_ENGINE_GS_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineGs : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineGs(const SmpObj &smp, const size_t n);

    ~SmpEngineGs() {
    }

    auto GetEngineName() const -> String override {
      return "Gs-Seq-CPU";
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

    // queue of free men
    Queue<int> free_men_queue_;

    Vector<int> next_proposed_w_;

    Vector<int> partner_rank_;
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_GS_H
