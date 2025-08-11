#include "smp/smp_engine_mw_1_f9.h"

#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineMwF9::SmpEngineMwF9(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_),
      partner_rank_(n_) {
  }


  void SmpEngineMwF9::InitProc() {
    // Init Rank Matrix
    // std::vector<std::thread> threads;
    //
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

    // Init Other Metadata
    for (int i = 0; i < n_; ++i) {
      partner_rank_[i] = n_;
      next_proposed_w_[i] = 0;
    }
  }

  void SmpEngineMwF9::CoreProc() {
    for (int m = 0; m < n_; m++) {
      // MWProcedure(m);
      int w_idx, m_rank, m_idx, w_rank, p_rank;
      m_idx = m;
      w_rank = this->next_proposed_w_[m_idx];
      bool is_matched = false;

      // auto flatten_pref_lists_m_ptr = RawPtr(smp_.flatten_pref_lists_m_vec);
      // auto flatten_pref_lists_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

      while (!is_matched) {
        w_idx = smp_.flatten_pref_lists_m_vec[m_idx * n_ + w_rank];
        // w_idx = smp_.flatten_pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
        // w_idx = flatten_pref_lists_m_ptr[IDX_MUL_ADD(m_idx, n_, w_rank)];

        // m_rank = this->rank_mtx_w_[w_idx * n_ + m_idx];
        m_rank = this->rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

        // std::cout << "Man " << m_idx << " proposed to " << w_rank << " th Woman "
        //           << w_idx;
        w_rank += 1;
        p_rank = this->partner_rank_[w_idx];
        if (p_rank == n_) {
          this->next_proposed_w_[m_idx] = w_rank;
          this->partner_rank_[w_idx] = m_rank;
          is_matched = true;
          // printf("and woman %d is unpaired, so STOP\n", w_idx);
        } else if (p_rank > m_rank) {
          this->next_proposed_w_[m_idx] = w_rank;
          this->partner_rank_[w_idx] = m_rank;

          m_idx = smp_.flatten_pref_lists_w_vec[w_idx * n_ + p_rank];
          // m_idx = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
          // m_idx = flatten_pref_lists_w_ptr[IDX_MUL_ADD(w_idx, n_, p_rank)];
          w_rank = this->next_proposed_w_[m_idx];
          // printf("and gets ACCEPTED. Since woman %d is paired with man %d, so
          // "SWITCH\n", w_idx, m_idx);
        } else {
          // printf("and gets REJECTED, so move to next woman\n");
        }
      }
    }
  }

  auto SmpEngineMwF9::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
