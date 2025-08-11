#pragma once

#include <atomic>
#include <queue>
#include <smp/smp.h>
#include <vector>

namespace bamboosmp {
  class GS {
  public:
    // Constructor
    GS(const SmpObj &smp, int thread_limit, int size);

    // GS(SmpObj *smp, std::vector<int> husband_rank_vector);
    ~GS();

    // rank matrix printer
    void InitGpu();

    void InitCpuSeq();

    void InitCpuPar();

    auto StartGS() -> float;

    auto GetMatchVector() const -> std::vector<int>;

    void PrintMatches();

    // GSParallel
    auto StartGSParallel() -> float;

    auto GetMatchVectorParallelCPU() const -> std::vector<int>;

  private:
    const SmpObj &smp_;
    int n_, num_blocks_, num_threads_per_block_;

    int *device_pref_lists_m_;
    int *device_pref_lists_w_;
    int *device_rank_mtx_w_;

    int *host_rank_mtx_w_;

    // queue of free men
    std::queue<int> free_men_queue_;

    int *next_proposed_w_;

    int *husband_rank_;

    std::vector<int> husband_rank_vector_;

    // parallel GS
    int max_num_threads_;
    std::vector<std::queue<int> > free_men_queues_;

    void initialize_queue(std::queue<int> &q, int start, int end);

    auto initialize_multiple_queues(int num_threads, int n) -> std::vector<std::queue<int> >;

    void GSParallelProcedure(int m);

    std::atomic<int> *atomic_partner_rank_;
  };
} // namespace bamboosmp
