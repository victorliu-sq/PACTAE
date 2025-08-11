#ifndef SMP_INIT_1_H
#define SMP_INIT_1_H

#include "smp.h"
#include "smp_init.h"
#include "smp/smp_init_abs.h"

namespace bamboosmp {
  class SmpInitEngine1 : public AbsSmpInitEngine {
  public:
    SmpInitEngine1(const SmpObjInit &smp, size_t n, bool init_pr_mtx)
      : AbsSmpInitEngine(init_pr_mtx),
        smp_(smp),
        n_(n),
        num_threads_(1),
        rank_mtx_w_(n_ * n_),
        pr_mtx_(n_ * n_) {
    }

    auto GetEngineName() const -> String override {
      return "Smp-Init-SingleCore";
    }

  protected:
    void InitRankMatrix() override {
      std::vector<std::thread> threads;

      for (size_t tid = 0; tid < num_threads_; ++tid) {
        threads.emplace_back([=] {
          size_t rows_per_thread = CEIL_DIV(this->n_, this->num_threads_);
          size_t start_row = tid * rows_per_thread;
          size_t end_row = (tid == num_threads_ - 1) ? n_ : start_row + rows_per_thread;
          for (size_t w_idx = start_row; w_idx < end_row; ++w_idx) {
            for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
              size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
              rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
            }
          }
        });
      }

      for (auto &th: threads) {
        th.join();
      }
    }

    void InitPRMatrix() override {
      std::vector<std::thread> threads;


      for (size_t tid = 0; tid < num_threads_; ++tid) {
        threads.emplace_back([=] {
          size_t rows_per_thread = CEIL_DIV(this->n_, this->num_threads_);
          size_t start_row = tid * rows_per_thread;
          size_t end_row = (tid == num_threads_ - 1) ? n_ : start_row + rows_per_thread;
          for (size_t m_idx = start_row; m_idx < end_row; ++m_idx) {
            for (size_t w_rank = 0; w_rank < n_; ++w_rank) {
              int w_idx = this->smp_.flatten_pref_lists_m_vec[IDX_MUL_ADD(m_idx, this->n_, w_rank)];
              int m_rank = this->rank_mtx_w_[IDX_MUL_ADD(w_idx, this->n_, m_idx)];
              this->pr_mtx_[IDX_MUL_ADD(m_idx, this->n_, w_rank)] = {w_idx, m_rank};
            }
          }
        });
      }

      for (auto &th: threads) {
        th.join();
      }
    }

  private:
    const SmpObjInit &smp_;
    const size_t n_;
    const int num_threads_;

    Vector<int> rank_mtx_w_;
    Vector<PRNode> pr_mtx_;
  };
}

#endif //SMP_INIT_1_H
