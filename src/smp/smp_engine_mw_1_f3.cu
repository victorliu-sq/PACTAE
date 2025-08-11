#include "smp/smp_engine_mw_1_f3.h"

#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineMw1F3::SmpEngineMw1F3(const SmpObj &smp, const size_t n)
    : smp_(smp),
      n_(n),
      num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
      // make rank_mtx large enough to use L3 / DRAM
      rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_),
      partner_rank_(n_) {
  }


  // void SmpEngineMw1F3::InitProc() {
  //   // Init Rank Matrix
  //   std::vector<std::thread> threads;
  //
  //   for (size_t tid = 0; tid < num_threads_; ++tid) {
  //     threads.emplace_back([=] {
  //       size_t rows_per_thread = CEIL_DIV(this->n_, this->num_threads_);
  //       size_t start_row = tid * rows_per_thread;
  //       size_t end_row = (tid == num_threads_ - 1) ? n_ : start_row + rows_per_thread;
  //
  //       if (start_row >= end_row) return; // no work for this thread
  //
  //       for (size_t w_idx = start_row; w_idx < end_row; ++w_idx) {
  //         for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
  //           size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
  //           rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
  //         }
  //       }
  //     });
  //   }
  //
  //   for (auto &th: threads) {
  //     th.join();
  //   }
  //
  //   // Init Other Metadata
  //   for (int i = 0; i < n_; ++i) {
  //     partner_rank_[i] = n_;
  //     next_proposed_w_[i] = 0;
  //   }
  // }

  void SmpEngineMw1F3::InitProc() {
    std::vector<std::thread> threads;

    const size_t max_threads = std::min<size_t>(n_, num_threads_); // <= n_ threads
    const size_t rows_per_thread = CEIL_DIV(this->n_, max_threads);

    for (size_t tid = 0; tid < max_threads; ++tid) {
      threads.emplace_back([=] {
        const size_t start_row = tid * rows_per_thread;
        const size_t end_row = std::min(this->n_, start_row + rows_per_thread);
        if (start_row >= end_row) return;

        for (size_t w_idx = start_row; w_idx < end_row; ++w_idx) {
          for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
            const size_t m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
            rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = static_cast<int>(m_rank);
          }
        }
      });
    }
    for (auto &th: threads) th.join();

    for (int i = 0; i < static_cast<int>(n_); ++i) {
      partner_rank_[i] = static_cast<int>(n_);
      next_proposed_w_[i] = 0;
    }
  }

  void SmpEngineMw1F3::CoreProc() {
    StopWatch sw(false); // for timeing random access;

    for (int m = 0; m < n_; m++) {
      // MWProcedure(m);
      int w_idx, m_rank, m_idx, w_rank, p_rank;
      m_idx = m;
      w_rank = this->next_proposed_w_[m_idx];
      bool is_matched = false;

      // auto flatten_pref_lists_m_ptr = RawPtr(smp_.flatten_pref_lists_m_vec);
      // auto flatten_pref_lists_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

      while (!is_matched) {
        // w_idx = smp_.flatten_pref_lists_m_[m_idx * n_ + w_rank];
        w_idx = smp_.flatten_pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
        // w_idx = flatten_pref_lists_m_ptr[IDX_MUL_ADD(m_idx, n_, w_rank)];

        sw.Start();
        // m_rank = this->rank_mtx_w_[w_idx * n_ + m_idx];
        m_rank = this->rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];
        sw.Stop();
        this->rand_access_time_ += sw.GetEclapsedNs();

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

          // m_idx = smp_.flatten_pref_lists_w_[w_idx * n_ + p_rank];
          m_idx = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)];
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

  auto SmpEngineMw1F3::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = partner_rank_[w_id];
      int m_id = smp_.flatten_pref_lists_w_vec[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }
    return match_vec;
  }
}
