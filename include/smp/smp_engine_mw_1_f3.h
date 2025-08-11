#ifndef SMP_ENGINE_MW_1_P_H
#define SMP_ENGINE_MW_1_P_H

#include "smp.h"
#include "smp_engine_abs.h"

namespace bamboosmp {
  class SmpEngineMw1F3 : public AbsSmpEngine {
  public:
    // Constructor
    SmpEngineMw1F3(const SmpObj &smp, const size_t n);

    ~SmpEngineMw1F3() = default;

    auto GetEngineName() const -> String override {
      return "Mw-Seq-CPU-Profile";
    }

    auto IsPerfect() const -> bool override {
      return false;
    }

    void InitProc() override;

    void CoreProc() override;

    auto PostProc() -> Matching override;

    void PrintProfilingInfo() override {
      std::cout << this->GetEngineName() << " Profiling Info:" << std::endl;
      std::cout << this->GetEngineName() << " Init Time: " << this->init_time_ << " ms" << std::endl;
      std::cout << this->GetEngineName() << " Core Time: " << this->core_time_ << " ms" << std::endl;
      std::cout << this->GetEngineName() << " Random Access Time: " << this->rand_access_time_ / 1e6 << " ms" <<
          std::endl;
      std::cout << this->GetEngineName() << " Total Time: " << this->init_time_ + this->core_time_ << " ms" <<
          std::endl;
      std::cout << "==================================================================" << std::endl;

      LOG(INFO) << this->GetEngineName() << " Profiling Info:";
      LOG(INFO) << this->GetEngineName() << " Init Time: " << this->init_time_ << " ms";
      LOG(INFO) << this->GetEngineName() << " Core Time: " << this->core_time_ << " ms";
      LOG(INFO) << this->GetEngineName() << " Random Access Time: " << this->rand_access_time_ / 1e6 << " ms";
      LOG(INFO) << this->GetEngineName() << " Total Time: " << this->init_time_ + this->core_time_ << " ms";
      LOG(INFO) << "==================================================================";
    }

  private:
    ns_t rand_access_time_{0};

    const SmpObj &smp_;
    const size_t n_;
    const int num_threads_;

    HVector<int> rank_mtx_w_;

    Vector<int> next_proposed_w_;

    Vector<int> partner_rank_;

    // void MWProcedure(int m);
  };
} // namespace bamboosmp

#endif //SMP_ENGINE_MW_1_P_H
