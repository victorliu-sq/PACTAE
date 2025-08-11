#include <set>
#include <smp/smp_engine_hybrid.cuh>
#include <utils/utils.h>
#include "utils/launcher.h"
#include <vector>
#include <glog/logging.h>
#include <cuda.h>
#include <utils/stopwatch.h>

#include "utils/array_view.h"

namespace bamboosmp {
  SmpEngineHybrid::SmpEngineHybrid(const SmpObj &smp, int size)
    : smp_(smp), n_(size) {
    cudaSetDevice(0);
    stable_matching_.resize(n_, 0);

    std::set<int> topChoicesSet;
    for (int m = 0; m < n_; ++m) {
      // int topChoice = smp_.flatten_pref_lists_m_[m * n_];
      int topChoice = smp_.flatten_pref_lists_m_vec[IDX_MUL(m, n_)];
      topChoicesSet.insert(topChoice);
      stable_matching_[m] = topChoice;
    }
    is_perfect_ = topChoicesSet.size() == n_;

    // Allocate Device Memory Space
    const size_t arr_size = static_cast<size_t>(n_);
    const size_t mtx_size = SIZE_MUL(n_, n_);

    StopWatch sw_init_alloc;
    host_partner_rank_.resize(arr_size, n_);
    host_next_proposed_w_.resize(arr_size, 0);

    host_rank_mtx_w_.resize(mtx_size);
    device_partner_rank_.resize(arr_size);
    device_next_proposed_w_.resize(arr_size);
    device_pref_lists_m_.resize(mtx_size);
    device_pref_lists_w_.resize(mtx_size);
    device_rank_mtx_w_.resize(mtx_size);
    device_prnodes_m_.resize(mtx_size);

    CUDA_CHECK(cudaMallocHost(&host_prnodes_m_view_, mtx_size * sizeof(PRNode), cudaHostAllocDefault));
    sw_init_alloc.Stop();
    std::cout << "SmpEngineHybrid::InitProc-Alloc spends time " << sw_init_alloc.GetEclapsedMs() << std::endl;
  }

  auto SmpEngineHybrid::GetStableMarriage() const -> std::vector<int> {
    std::vector<int> result(n_);
    for (int i = 0; i < n_; i++) {
      result[i] = stable_matching_[i];
    }
    return result;
  }

  void SmpEngineHybrid::PrintMatching() const {
    LOG(INFO) << "Stable Matching:";
    for (int i = 0; i < n_; ++i) {
      LOG(INFO) << "(Man:" << i << " is paired with Woman:" << stable_matching_[i] << ")";
    }
  }

  void SmpEngineHybrid::PostProc() {
    LOG(INFO) << "SmpEngineHybrid: PostProc Starts";
    for (int w = 0; w < n_; w++) {
      int m = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w, n_, host_partner_rank_[w])];
      stable_matching_[m] = w;
    }
    LOG(INFO) << "SmpEngineHybrid: PostProc Completes";
  }

  /* ------------------------------------- Init ------------------------------------------------ */
  /* ------------------------------------------------------------------------------------------------ */
  void SmpEngineHybrid::InitProc() {
    auto start_hybrid_init = getNanoSecond();
    const size_t mtx_size = SIZE_MUL(n_, n_);

    StopWatch sw_init_h2d;
    thrust::copy(host_partner_rank_.begin(),
                 host_partner_rank_.end(),
                 device_partner_rank_.begin());

    thrust::copy(host_next_proposed_w_.begin(),
                 host_next_proposed_w_.end(),
                 device_next_proposed_w_.begin());

    thrust::copy(smp_.flatten_pref_lists_m_vec.begin(),
                 smp_.flatten_pref_lists_m_vec.end(),
                 device_pref_lists_m_.begin());

    thrust::copy(smp_.flatten_pref_lists_w_vec.begin(),
                 smp_.flatten_pref_lists_w_vec.end(),
                 device_pref_lists_w_.begin());
    sw_init_h2d.Stop();
    std::cout << "SmpEngineHybrid::InitProc-H2D spends time " << sw_init_h2d.GetEclapsedMs() << std::endl;

    pref_lists_w_view_ = RawPtr(smp_.flatten_pref_lists_w_vec);

    StopWatch sw_init_rank_mtx_kernel;
    LaunchKernelForEach(default_stream_, mtx_size,
                        InitRankMatrixKernelTasklet{
                          n_,
                          ConstArrayView<int>(device_pref_lists_w_),
                          ArrayView<int>(device_rank_mtx_w_)
                        });
    sw_init_rank_mtx_kernel.Stop();
    std::cout << "SmpEngineHybrid::InitProc-InitRankMatrix spends time " << sw_init_rank_mtx_kernel.GetEclapsedMs() <<
        std::endl;

    StopWatch sw_init_prmatrix_kernel;
    LaunchKernelForEach(default_stream_, mtx_size,
                        InitPRMatrixKernelTasklet{
                          n_,
                          ConstArrayView<int>(device_rank_mtx_w_),
                          ConstArrayView<int>(device_pref_lists_m_),
                          ArrayView<PRNode>(device_prnodes_m_)
                        });
    sw_init_prmatrix_kernel.Stop();
    std::cout << "SmpEngineHybrid::InitProc-InitPRMatrix spends time " << sw_init_prmatrix_kernel.GetEclapsedMs() <<
        std::endl;

    LOG(INFO) << "Initialization procedure completed.";
    auto stop_hybrid_init = getNanoSecond();
    std::cout << "[SmpHyrid:InitProc] Init takes time " << (stop_hybrid_init - start_hybrid_init) / 1e6 << " ms." <<
        std::endl;
  }

  auto SmpEngineHybrid::RetrievePRMatrixForTest() -> Vector<PRNode> {
    Vector<PRNode> prmatrix(SIZE_MUL(n_, n_));
    thrust::copy(device_prnodes_m_.begin(),
                 device_prnodes_m_.end(),
                 prmatrix.begin());
    return prmatrix;
  }

  auto SmpEngineHybrid::RetrieveRankMatrixForTest() -> Vector<int> {
    Vector<int> host_rank_mtx(SIZE_MUL(n_, n_));
    thrust::copy(device_rank_mtx_w_.begin(),
                 device_rank_mtx_w_.end(),
                 host_rank_mtx.begin());
    return host_rank_mtx;
  }


  /* ------------------------------------- Core Computation ------------------------------------- */

  void SmpEngineHybrid::CoreProc() {
    std::thread thread_gpu(&SmpEngineHybrid::DoWorkOnGpu, this);
    auto start_cpu_work = getNanoSecond();
    this->DoWorkOnCpu();
    auto stop_cpu_work = getNanoSecond();
    std::cout << "[Hybrid::DoCPUWork] takes time: " << (stop_cpu_work - start_cpu_work) / 1e6 << " ms." << std::endl;
    // thread_gpu.join();
    thread_gpu.detach();
  }

  void SmpEngineHybrid::DoWorkOnCpu() {
    int unmatched_id, unmatched_num;

    MonitorProceture(unmatched_id, unmatched_num, monitor_stream_);
    // exit when unmatched_num is either 0 or 1.

    if (unmatched_num == 1) {
      StopWatch timer_start_prnodes_d2h;
      CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_view_,
        RawPtr(device_prnodes_m_),
        SIZE_MUL(n_, n_) * sizeof(PRNode),
        cudaMemcpyDeviceToHost,
        monitor_stream_.cuda_stream()));
      monitor_stream_.Sync();
      timer_start_prnodes_d2h.Stop();
      std::cout << "[Hybdird] PRNOdes D2H spends time: " << timer_start_prnodes_d2h.GetEclapsedMs() << std::endl;

      LOG(INFO) << "CPU starts LAProcedure.";
      auto start_la = getNanoSecond();
      LAProcedure(unmatched_id);
      auto stop_la = getNanoSecond();
      std::cout << "[Hybrid::LAProcedure] takes time: " << (stop_la - start_la) / 1e6 << " ms." << std::endl;

      LOG(INFO) << "CheckKernel (CPU) won the contention.";
    } else {
      LOG(INFO) << "CheckKernel (CPU) won the contention.";
    }
    // final_partner_rank_ = host_partner_rank_;
  }

  void SmpEngineHybrid::MonitorProceture(int &unmatched_id,
                                         int &unmatched_num,
                                         const CudaStream &monitor_stream) {
    int it = 0;
    // const int total = n_ * (n_ - 1) / 2;
    const size_t total = SIZE_MUL(n_, (n_ - 1)) / 2;
    do {
      // timer_start(true);
      // timer_next("Copy Rank D2H");
      SLEEP_MILLISECONDS(1);

      CUDA_CHECK(cudaMemcpyAsync(RawPtr(host_partner_rank_),
        RawPtr(device_partner_rank_),
        static_cast<size_t>(n_) * sizeof(int),
        cudaMemcpyDeviceToHost,
        monitor_stream.cuda_stream()));

      unmatched_id = total;
      unmatched_num = 0;

      // timer_next("Iterate N Ranks");
      for (int w = 0; w < n_; w++) {
        if (host_partner_rank_[w] == n_) {
          unmatched_num++;
        } else {
          int m_rank = host_partner_rank_[w];
          // unmatched_id -= pref_lists_w_view_[w * n_ + m_rank];
          unmatched_id -= pref_lists_w_view_[IDX_MUL_ADD(w, n_, m_rank)];
        }
      }
      // timer_end();
      // LOG(INFO) << "Iteration " << it << ": # of unmatched men = " << unmatched_num;
      it++;
    } while (unmatched_num > 1);
  }

  void SmpEngineHybrid::LAProcedure(int m) {
    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node;
    bool is_matched = false;
    int iterations = 0;
    while (!is_matched) {
      iterations += 1;
      // temp_node = host_prnodes_m_view_[m_idx * n_ + w_rank];
      temp_node = host_prnodes_m_view_[IDX_MUL_ADD(m_idx, n_, w_rank)];
      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      p_rank = host_partner_rank_[w_idx];
      if (p_rank == n_) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;
        is_matched = true;
      } else if (p_rank > m_rank) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;

        // m_idx = pref_lists_w_view_[w_idx * n_ + p_rank];
        m_idx = pref_lists_w_view_[IDX_MUL_ADD(w_idx, n_, p_rank)];
        w_rank = host_next_proposed_w_[m_idx];
      } else {
        w_rank++;
      }
    }
  }

  void SmpEngineHybrid::DoWorkOnGpu() {
    PROFILE_SCOPE("BambooSMP-GPUExec");
    LaunchKernelForEach(default_stream_, n_,
                        LAKernelTasklet{
                          n_,
                          ConstArrayView<PRNode>(device_prnodes_m_),
                          ConstArrayView<int>(device_pref_lists_w_),
                          ArrayView<int>(device_partner_rank_),
                          ArrayView<int>(device_next_proposed_w_)
                        });
  }
} // namespace bamboosmp
