#include "smp/smp_engine_gs_2.h"
#include "utils/stopwatch.h"

namespace bamboosmp {
  SmpEngineGs2::SmpEngineGs2(const SmpObj &smp, size_t n)
    : smp_(smp),
      n_(n),
      num_threads_(std::min(std::thread::hardware_concurrency(), 96u)),
      rank_mtx_w_(n_ * n_),
      next_proposed_w_(n_),
      free_men_queues_(n_) {
    atomic_partner_rank_ = new std::atomic<int>[n_];
  }

  SmpEngineGs2::~SmpEngineGs2() {
    delete[] atomic_partner_rank_;
  }

  void SmpEngineGs2::InitProc() {
    Vector<std::thread> threads;
    for (size_t tid = 0; tid < num_threads_; ++tid) {
      threads.emplace_back([=] {
        size_t rows_per_thread = CEIL_DIV(this->n_, this->num_threads_);
        size_t start_row = tid * rows_per_thread;
        size_t end_row = (tid == num_threads_ - 1) ? n_ : start_row + rows_per_thread;
        for (size_t w_idx = start_row; w_idx < end_row; ++w_idx) {
          for (size_t m_rank = 0; m_rank < n_; ++m_rank) {
            size_t m_idx = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, m_rank)];
            rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
          }
        }
      });
    }

    for (auto &th: threads) {
      th.join();
    }

    for (int i = 0; i < n_; ++i) {
      next_proposed_w_[i] = 0;
      atomic_partner_rank_[i].store(n_);
    }

    this->InitQueues();
  }

  void SmpEngineGs2::InitQueues() {
    int avg_num_men = this->n_ / this->num_threads_;

    for (int i = 0; i < this->num_threads_; ++i) {
      int start = i * avg_num_men;
      int end = (i != this->num_threads_ - 1) ? (i + 1) * avg_num_men : this->n_;

      for (int j = start; j < end; ++j) {
        this->free_men_queues_[i].push(j);
        // std::cout << "Add man " << j << "into Queue " << i << std::endl;
      }
    }
  }

  void SmpEngineGs2::CoreProc() {
    Vector<std::thread> threads;
    threads.reserve(this->num_threads_);

    // Launch threads in batches, each handling one distinct woman (0 to n_-1)
    for (int tid = 0; tid < this->num_threads_; ++tid) {
      threads.emplace_back(&SmpEngineGs2::GsParTasklet, this, tid);
    }

    for (auto &thread: threads) {
      thread.join();
    }
  }

  void SmpEngineGs2::GsParTasklet(int tid) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    auto free_men_queue = free_men_queues_[tid];

    m_idx = free_men_queue.front();
    free_men_queue.pop();
    w_rank = 0;

    StopWatch sw(false); // for Equality Comparison;

    bool done = false;
    while (!done) {
      w_idx = smp_.flatten_pref_lists_m[IDX_MUL_ADD(m_idx, n_, w_rank)];
      m_rank = rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)];

      w_rank += 1;
      p_rank = atomic_partner_rank_[w_idx].load();
      if (p_rank < m_rank) {
        continue;
      }

      while (m_rank < p_rank) {
        int expected = p_rank;
        if (atomic_partner_rank_[w_idx].compare_exchange_strong(expected,
                                                                m_rank)) {
          next_proposed_w_[m_idx] = w_rank;
          if (p_rank != n_) {
            int new_free_man = smp_.flatten_pref_lists_w[IDX_MUL_ADD(w_idx, n_, p_rank)]; free_men_queue.push(new_free_man);
          }
          if (free_men_queue.size() > 0) {
            m_idx = free_men_queue.front();

            free_men_queue.pop();
            w_rank = next_proposed_w_[m_idx];
          } else {
            done = true;
          }
          p_rank = m_rank;
        } else {
          p_rank = expected;
        }
      }
    }
  }

  auto SmpEngineGs2::PostProc() -> Matching {
    Matching match_vec(n_);

    for (int w_id = 0; w_id < match_vec.size(); w_id++) {
      int m_rank = atomic_partner_rank_[w_id].load();
      int m_id = smp_.flatten_pref_lists_w[w_id * n_ + m_rank];
      match_vec[m_id] = w_id;
    }

    return match_vec;
  }
}
