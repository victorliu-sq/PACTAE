#ifndef SMP_ENGINE_ABS_H
#define SMP_ENGINE_ABS_H
#include "utils/stopwatch.h"
#include "utils/types.h"

class AbsSmpEngine {
public:
  virtual ~AbsSmpEngine() = default;

  auto FindStableMatching() -> Matching {
    this->sw_.Start();
    bool is_perfect = this->IsPerfect();
    this->sw_.Stop();
    this->precheck_time_ = this->sw_.GetEclapsedMs();

    if (!is_perfect) {
      // Init Phase
      this->sw_.Start();
      this->InitProc(); // Do the computation
      this->sw_.Stop();
      this->init_time_ = this->sw_.GetEclapsedMs();

      // Core Computation Phase
      this->sw_.Start();
      this->CoreProc(); // Do the computation
      this->sw_.Stop();
      this->core_time_ = this->sw_.GetEclapsedMs();
    } else {
      std::cout << "Perfect matching identified; skipping execution steps." << std::endl;
    }

    // Post-Process
    return this->PostProc();
  }

  virtual void PrintProfilingInfo() {
    std::cout << this->GetEngineName() << " Profiling Info:" << std::endl;
    std::cout << this->GetEngineName() << " Precheck Time: " << this->precheck_time_ << " ms" << std::endl;
    std::cout << this->GetEngineName() << " Init Time: " << this->init_time_ << " ms" << std::endl;
    std::cout << this->GetEngineName() << " Core Time: " << this->core_time_ << " ms" << std::endl;
    std::cout << this->GetEngineName() << " Total Time: " << this->precheck_time_ + this->init_time_ + this->core_time_ << " ms" << std::endl;
    std::cout << "==================================================================" << std::endl;

    LOG(INFO) << this->GetEngineName() << " Profiling Info:";
    LOG(INFO) << this->GetEngineName() << " Precheck Time: " << this->precheck_time_ << " ms";
    LOG(INFO) << this->GetEngineName() << " Init Time: " << this->init_time_ << " ms";
    LOG(INFO) << this->GetEngineName() << " Core Time: " << this->core_time_ << " ms";
    LOG(INFO) << this->GetEngineName() << " Total Time: " << this->precheck_time_ +  this->init_time_ + this->core_time_ << " ms";
    LOG(INFO) << "==================================================================";
  }

protected:
  virtual auto IsPerfect() const -> bool = 0;

  virtual void InitProc() = 0;

  virtual void CoreProc() = 0;

  virtual auto PostProc() -> Matching = 0;

  virtual auto GetEngineName() const -> String = 0;

  ms_t precheck_time_{0};
  ms_t init_time_{0};
  ms_t core_time_{0};

private:
  StopWatch sw_{false};
};

#endif //SMP_ENGINE_ABS_H
