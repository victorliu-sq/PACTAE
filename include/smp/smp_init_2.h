#ifndef SMP_INIT_2_H
#define SMP_INIT_2_H

#include "smp/smp_init_abs.h"

namespace bamboosmp {
  class SmpInitEngine2 : public AbsSmpInitEngine {
  public:
    SmpInitEngine2(const SmpObjInit &smp, size_t n, bool init_pr_mtx)
      : AbsSmpInitEngine(init_pr_mtx),
        smp_(smp),
        n_(n),
        num_threads_(std::min(std::thread::hardware_concurrency(), 8u)),
        rank_mtx_w_(n_ * n_),
        pr_mtx_(n_ * n_) {
    }

    auto GetEngineName() const -> String override {
      return "Smp-Init-MultiCore";
    }

  protected:
    // void InitRankMatrix() override {
    //   const size_t n = this->n_;
    //   const size_t max_threads = static_cast<size_t>(this->num_threads_);
    //   std::vector<std::thread> threads;
    //   threads.reserve(max_threads);
    //
    //   for (size_t w_idx = 0; w_idx < n; ++w_idx) {
    //     // launch one thread for this row
    //     threads.emplace_back([this, n, w_idx] {
    //       for (size_t m_rank = 0; m_rank < n; ++m_rank) {
    //         const size_t m_idx =
    //             static_cast<size_t>(this->smp_.flatten_pref_lists_w_vec
    //               [IDX_MUL_ADD(w_idx, n, m_rank)]);
    //         this->rank_mtx_w_[IDX_MUL_ADD(w_idx, n, m_idx)] =
    //             static_cast<int>(m_rank);
    //       }
    //     });
    //
    //     // if we've reached the cap, wait for all to finish before continuing
    //     if (threads.size() == max_threads) {
    //       for (auto &th: threads) th.join();
    //       threads.clear();
    //     }
    //   }
    //
    //   // join any remaining threads from the last partial batch
    //   for (auto &th: threads) th.join();
    // }
    //
    // void InitPRMatrix() override {
    //   const size_t n = this->n_;
    //   const size_t max_threads = static_cast<size_t>(this->num_threads_);
    //   std::vector<std::thread> threads;
    //   threads.reserve(max_threads);
    //
    //   for (size_t m_idx = 0; m_idx < n; ++m_idx) {
    //     threads.emplace_back([this, n, m_idx] {
    //       for (size_t w_rank = 0; w_rank < n; ++w_rank) {
    //         const size_t pr_id = IDX_MUL_ADD(m_idx, n, w_rank);
    //
    //         const int w_idx =
    //             this->smp_.flatten_pref_lists_m_vec[pr_id];
    //         const int m_rank =
    //             this->rank_mtx_w_[IDX_MUL_ADD(static_cast<size_t>(w_idx), n, m_idx)];
    //
    //         this->pr_mtx_[pr_id] = PRNode{w_idx, m_rank};
    //       }
    //     });
    //
    //     if (threads.size() == max_threads) {
    //       for (auto &th: threads) th.join();
    //       threads.clear();
    //     }
    //   }
    //
    //   for (auto &th: threads) th.join();
    // }

    void InitRankMatrix() override {
      const size_t n = this->n_;
      const size_t max_threads = std::max<size_t>(1, std::min(n, (size_t) this->num_threads_));

      std::vector<std::thread> threads(max_threads); // fixed window

      for (size_t w_idx = 0; w_idx < n; ++w_idx) {
        const size_t slot = w_idx % max_threads;

        // if this slot is already running a thread, join it before reusing
        if (w_idx >= max_threads && threads[slot].joinable()) {
          threads[slot].join();
        }

        threads[slot] = std::thread([this, n, w_idx] {
          for (size_t m_rank = 0; m_rank < n; ++m_rank) {
            const size_t m_idx =
                static_cast<size_t>(this->smp_.flatten_pref_lists_w_vec
                  [IDX_MUL_ADD(w_idx, n, m_rank)]);
            this->rank_mtx_w_[IDX_MUL_ADD(w_idx, n, m_idx)] =
                static_cast<int>(m_rank);
          }
        });
      }

      // join remaining live threads
      const size_t live = std::min(n, max_threads);
      for (size_t i = 0; i < live; ++i) {
        if (threads[i].joinable()) threads[i].join();
      }
    }

    void InitPRMatrix() override {
      const size_t n = this->n_;
      const size_t max_threads = std::max<size_t>(1, std::min(n, (size_t) this->num_threads_));

      std::vector<std::thread> threads(max_threads); // fixed window

      for (size_t m_idx = 0; m_idx < n; ++m_idx) {
        const size_t slot = m_idx % max_threads;

        if (m_idx >= max_threads && threads[slot].joinable()) {
          threads[slot].join();
        }

        threads[slot] = std::thread([this, n, m_idx] {
          for (size_t w_rank = 0; w_rank < n; ++w_rank) {
            const size_t pr_id = IDX_MUL_ADD(m_idx, n, w_rank);

            const int w_idx =
                this->smp_.flatten_pref_lists_m_vec[pr_id];
            const int m_rank =
                this->rank_mtx_w_[IDX_MUL_ADD(static_cast<size_t>(w_idx), n, m_idx)];

            this->pr_mtx_[pr_id] = PRNode{w_idx, m_rank};
          }
        });
      }

      const size_t live = std::min(n, max_threads);
      for (size_t i = 0; i < live; ++i) {
        if (threads[i].joinable()) threads[i].join();
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

#endif //SMP_INIT_2_H
