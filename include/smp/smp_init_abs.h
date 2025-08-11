#ifndef SMP_INIT_ABS_H
#define SMP_INIT_ABS_H

#include "utils/stopwatch.h"
#include "utils/types.h"

namespace bamboosmp {
  class AbsSmpInitEngine {
  public:
    AbsSmpInitEngine(bool init_pr_mtx)
      : init_pr_mtx_(init_pr_mtx) {
    }

    virtual ~AbsSmpInitEngine() = default;

    void Init() {
      this->sw_.Start();
      this->InitRankMatrix();
      this->sw_.Stop();
      this->init_rank_mtx_time_ = this->sw_.GetEclapsedMs();

      if (init_pr_mtx_) {
        this->sw_.Start();
        this->InitPRMatrix();
        this->sw_.Stop();
        this->init_pr_mtx_time_ = this->sw_.GetEclapsedMs();
      }
    }

    virtual void PrintProfilingInfo() {
      this->total_init_time_ = this->init_rank_mtx_time_;
      if (this->init_pr_mtx_) {
        this->total_init_time_ += init_pr_mtx_time_;
      }

      // Title
      OutStringStream oss_title;
      oss_title << this->GetEngineName();
      if (this->init_pr_mtx_) {
        oss_title << " Initialize Both RankMatrix and PRMatrix ";
      } else {
        oss_title << " Initialize RankMatrix Only ";
      }
      oss_title << "Profiling Info: ";
      String title = oss_title.str();

      std::cout << title << std::endl;
      std::cout << this->GetEngineName() << " Init Time: " << this->total_init_time_ << " ms" << std::endl;
      std::cout << "==================================================================" << std::endl;

      LOG(INFO) << title;
      LOG(INFO) << this->GetEngineName() << " Init Time: " << this->total_init_time_ << " ms";
      LOG(INFO) << "==================================================================";
    }

  protected:
    virtual auto GetEngineName() const -> String = 0;

    virtual void InitRankMatrix() = 0;

    virtual void InitPRMatrix() = 0;

    ms_t total_init_time_{0};
    ms_t init_rank_mtx_time_{0};
    ms_t init_pr_mtx_time_{0};

    bool init_pr_mtx_;

  private:
    StopWatch sw_{false};
  };
}

#endif //SMP_INIT_ABS_H
