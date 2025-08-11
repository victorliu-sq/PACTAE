#include "smp/smp_engine_mw_2_f9.h"

namespace bamboosmp {
  SmpEngineMw2F9::SmpEngineMw2F9(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
      rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_) {
    atomic_partner_rank_ = new std::atomic<int>[n_];
  }

  SmpEngineMw2F9::~SmpEngineMw2F9() {
    delete[] atomic_partner_rank_;
  }

  void SmpEngineMw2F9::InitProc() {
    // Vector<std::thread> threads;
    // for (size_t tid = 0; tid < num_threads_; ++tid) {
    //   threads.emplace_back([=] {
    //     size_t rows_per_thread = CEIL_DIV(this->n_, this->num_threads_);
    //     size_t start_row = tid * rows_per_thread;
    //     size_t end_row = (tid == num_threads_ - 1) ? n_ : start_row + rows_per_thread;
    //     for (size_t w_idx = start_row; w_idx < end_row; ++w_idx) {
    //       for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
    //         size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
    //         rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
    //       }
    //     }
    //   });
    // }
    //
    // for (auto &th: threads) {
    //   th.join();
    // }

    for (size_t w_idx = 0; w_idx < n_; ++w_idx) {
      for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
        // size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
        size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
        rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
      }
    }

    for (int i = 0; i < n_; ++i) {
      next_proposed_w_[i] = 0;
      atomic_partner_rank_[i].store(n_);
    }
  }

  void SmpEngineMw2F9::CoreProc() {
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads_; ++i) {
      // threads.emplace_back(&SmpEngineMwParC::MWParTasklet, this, i);
      threads.emplace_back([=](int tid) {
        size_t men_per_thread = CEIL_DIV(this->n_, this->num_threads_);
        size_t start_man = tid * men_per_thread;
        size_t end_man = std::min(this->n_, start_man + men_per_thread);
        for (int start_mi = start_man; start_mi < end_man; ++start_mi) {
          int w_idx, m_rank, m_idx, w_rank, p_rank;
          m_idx = start_mi;

          w_rank = this->next_proposed_w_[m_idx];
          bool is_married = false;
          while (!is_married) {
            // Figure 9
            // w_idx = this->smp_.flatten_pref_lists_m_vec[m_idx * n_ + w_rank];
            w_idx = this->smp_.flatten_pref_lists_m_vec[m_idx * n_ + w_rank];

            m_rank = this->rank_mtx_w_[w_idx * n_ + m_idx];

            w_rank += 1;
            p_rank = this->atomic_partner_rank_[w_idx].load();
            if (p_rank < m_rank) {
              continue;
            }

            // p_rank = this->atomic_partner_rank_[w_idx].load();
            while (m_rank < p_rank) {
              int expected = p_rank;
              if (this->atomic_partner_rank_[w_idx].compare_exchange_strong(expected, m_rank)) {
                this->next_proposed_w_[m_idx] = w_rank;
                if (expected == n_) {
                  is_married = true;
                } else {
                  // Figure 9
                  m_idx = this->smp_.flatten_pref_lists_w_vec[w_idx * n_ + expected];
                  // m_idx = this->smp_.flatten_pref_lists_w[w_idx * n_ + expected];
                  w_rank = this->next_proposed_w_[m_idx];
                }
                break;
              } else {
                p_rank = expected;
              }
            }
          }
        }
      }, i);
    }

    for (auto &thread: threads) {
      thread.join();
    }
  }

  auto SmpEngineMw2F9::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = atomic_partner_rank_[w_id].load();
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }

    return match_vec;
  }
}
