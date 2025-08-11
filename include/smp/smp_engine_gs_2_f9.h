#ifndef SMP_ENGINE_GS_2_F9_H
#define SMP_ENGINE_GS_2_F9_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineGs2F9 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineGs2F9(const SmpObj &smp, size_t n);

    ~SmpEngineGs2F9();

    auto GetEngineName() const -> String override {
      return "Gs-Par-Cpu:InitSeqCPU+ExecParCPU";
    }

    auto IsPerfect() const -> bool override {
      return false;
    }

    void InitProc() override;

    void CoreProc() override;

    auto PostProc() -> Matching override;

  private:
    void InitQueues();

    void GsParTasklet(int m);

    const SmpObj &smp_;
    const size_t n_;
    const int num_threads_;

    Vector<int> rank_mtx_w_;

    Vector<int> next_proposed_w_;

    // Vector<int> husband_rank_;
    // Vector<std::atomic<int> > atomic_partner_rank_;
    std::atomic<int> *atomic_partner_rank_;

    // parallelSmpEngineGsParCpu
    // Queue<int> free_men_queue_;
    Vector<Queue<int> > free_men_queues_;
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_GS_2_F9_H
