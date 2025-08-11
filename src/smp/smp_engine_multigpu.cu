#include "smp/smp_engine_multigpu.h"
#include <set>
#include <smp/kernel.h>
#include <utils/launcher.h>
#include <glog/logging.h>
#include <utils/stopwatch.h>

namespace bamboosmp {
  SmpEngineMultiGpu::SmpEngineMultiGpu(SmpObj &smp, int size, int num_gpus)
    : smp_(smp),
      n_(size),
      num_gpus_(num_gpus),
      multigpu_device_pref_lists_m_(0),
      multigpu_device_pref_lists_w_(0),
      multigpu_device_rank_mtx_w_(0),
      multigpu_device_prnodes_m_(0),
      multigpu_device_next_proposed_w_(0),
      multigpu_device_partner_rank_(0),
      tflag_(MakeSPtr<AtomicTFlag>()),
      host_prnodes_m_ptr_(nullptr) {
    int actual_gpu_count = 0;
    cudaGetDeviceCount(&actual_gpu_count);
    LOG(INFO) << "Available GPUs: " << actual_gpu_count;
    CHECK(num_gpus_ <= actual_gpu_count);

    // Each CUDA stream must be created after cudaSetDevice() is called for its GPU.
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      gpu_streams_.emplace_back();
      monitor_streams_.emplace_back();
    }

    stable_matching_.resize(n_, 0);
    std::set<int> topChoicesSet;
    for (int m = 0; m < n_; ++m) {
      // int topChoice = smp_.flatten_pref_lists_m_[m * n_];
      int topChoice = smp_.flatten_pref_lists_m_vec[IDX_MUL(m, n_)];
      topChoicesSet.insert(topChoice);
      stable_matching_[m] = topChoice;
    }
    is_perfect_ = topChoicesSet.size() == n_;

    // **INSERT THIS LOGGING CODE HERE**
    cudaSetDevice(0);
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    LOG(INFO) << "[End of constructor] GPU 0 memory: " << free_bytes / (1024 * 1024)
        << " MB free out of " << total_bytes / (1024 * 1024) << " MB.";
  }

  SmpEngineMultiGpu::~SmpEngineMultiGpu() {
    WriteTimingsToFile("data/benchmark/solo_scaling_timing_results.txt",
                       "SOLO", num_gpus_, n_);
    /* Temporarlity disable these for solo cases.
    LOG(INFO) << "Cleaning up resources in ~SmpEngineMultiGpu";
    this->DisablePeerAccess();

    // Free host pinned memory
    if (host_prnodes_m_ptr_ != nullptr) {
      // CUDA_CHECK(cudaFreeHost(host_prnodes_m_ptr_));
      // free(host_prnodes_m_ptr_);
      host_prnodes_m_.clear();
      host_prnodes_m_ptr_ = nullptr;
    }

    // Clear vectors explicitly (device memory)
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      gpu_streams_[gpu_id].Sync(); // Sync the GPU streams explicitly
      monitor_streams_[gpu_id].Sync(); // Sync monitor streams explicitly
      CUDA_CHECK(cudaDeviceSynchronize());

      multigpu_device_pref_lists_m_[gpu_id].clear();
      multigpu_device_pref_lists_w_[gpu_id].clear();
      multigpu_device_rank_mtx_w_[gpu_id].clear();
      multigpu_device_prnodes_m_[gpu_id].clear();
      multigpu_device_next_proposed_w_[gpu_id].clear();
      multigpu_device_partner_rank_[gpu_id].clear();

      // Destroy CUDA streams
      gpu_streams_[gpu_id].Destroy();
      monitor_streams_[gpu_id].Destroy();
    }
    */
  }


  // Method to disable peer access (reverse of EnablePeerAccess)
  void SmpEngineMultiGpu::DisablePeerAccess() {
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      CUDA_CHECK_CONTINUE(cudaSetDevice(gpu_id));
      for (int peer_gpu = 0; peer_gpu < num_gpus_; ++peer_gpu) {
        if (gpu_id != peer_gpu) {
          int can_access = 0;
          CUDA_CHECK_CONTINUE(cudaDeviceCanAccessPeer(&can_access, gpu_id, peer_gpu));
          if (can_access) {
            cudaError_t err = cudaDeviceDisablePeerAccess(peer_gpu);
            if (err == cudaSuccess) {
              LOG(INFO) << "Peer access disabled: GPU " << gpu_id << " -> GPU " << peer_gpu;
            } else if (err == cudaErrorPeerAccessNotEnabled) {
              LOG(INFO) << "Peer access not enabled, skipping disable: GPU " << gpu_id << " -> GPU " << peer_gpu;
            } else {
              LOG(ERROR) << "Failed to disable peer access GPU " << gpu_id << " -> GPU " << peer_gpu
                  << ": " << cudaGetErrorString(err);
            }
          }
        }
      }
    }
  }

  /* ------------------------------------- InitProc ------------------------------------------------ */
  /* ------------------------------------------------------------------------------------------------ */
  void SmpEngineMultiGpu::InitProc() {
    auto start_ns = getNanoSecond();
    LOG(INFO) << "SmpEngineMultiGpu::InitProc Starts";
    // Enabel access for each gpu to all its peers
    this->EnablePeerAccess();

    // Async Copy data H2D
    this->H2DAsync();

    // Launch Kernel Async
    this->InitMatriceAsync();


    LOG(INFO) << "SmpEngineMultiGpu::InitProc Completes";

    auto end_ns = getNanoSecond();
    std::cout << "[InitProc] total took " << (end_ns - start_ns) / 1e6 << " ms.";
  }

  void SmpEngineMultiGpu::EnablePeerAccess() {
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      for (int peer_gpu = 0; peer_gpu < num_gpus_; ++peer_gpu) {
        if (gpu_id != peer_gpu) {
          int can_access = 0;
          CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, gpu_id, peer_gpu));
          if (can_access) {
            cudaError_t err = cudaDeviceEnablePeerAccess(peer_gpu, 0);
            if (err == cudaSuccess) {
              LOG(INFO) << "Peer access explicitly enabled: GPU " << gpu_id << " -> GPU " << peer_gpu;
            } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
              LOG(INFO) << "Peer access already explicitly enabled: GPU " << gpu_id << " -> GPU " << peer_gpu;
            } else {
              LOG(FATAL) << "Failed to explicitly enable peer access GPU " << gpu_id << " -> GPU " << peer_gpu
                  << ": " << cudaGetErrorString(err);
            }
          } else {
            LOG(FATAL) << "Peer access not explicitly supported: GPU " << gpu_id << " -> GPU " << peer_gpu;
          }
        }
      }
    }
  }


  void SmpEngineMultiGpu::H2DAsync() {
    LOG(INFO) << "Entering AsyncCopyToGpus";

    final_partner_ranks_.resize(n_, n_);
    // host_prnodes_m_.resize(n_ * n_);
    host_next_proposed_w_.resize(n_, 0); {
      PROFILE_SCOPE("InitH2D");

      // Explicitly reserve vector sizes
      multigpu_device_next_proposed_w_.reserve(num_gpus_);
      multigpu_device_partner_rank_.reserve(num_gpus_);
      multigpu_device_pref_lists_m_.reserve(num_gpus_);
      multigpu_device_pref_lists_w_.reserve(num_gpus_);
      multigpu_device_rank_mtx_w_.reserve(num_gpus_);
      multigpu_device_prnodes_m_.reserve(num_gpus_);


      cudaSetDevice(0);
      size_t free_bytes_before, total_bytes_before;
      cudaMemGetInfo(&free_bytes_before, &total_bytes_before);
      LOG(INFO) << "[Before allocations] GPU 0 memory: " << free_bytes_before / (1024 * 1024) << " MB free of "
          << total_bytes_before / (1024 * 1024) << " MB.";

      FOR_EACH_GPU(num_gpus_, gpu_id) {
        int current_device;
        cudaGetDevice(&current_device);
        LOG(INFO) << "[BEFORE ALLOC] Currently on GPU: " << current_device;

        int arr_offset = PARTITION_START(n_, num_gpus_, gpu_id);
        int arr_size = PARTITION_SIZE(n_, num_gpus_, gpu_id);

        LOG(INFO) << "GPU: " << gpu_id
            << ", local_offset: " << arr_offset << ", local_size: " << arr_size;

        multigpu_device_next_proposed_w_.emplace_back(arr_size);
        multigpu_device_partner_rank_.emplace_back(arr_size);

        // 1-d array init
        CHECK(multigpu_device_next_proposed_w_.size() > gpu_id);
        CHECK(multigpu_device_partner_rank_.size() > gpu_id);

        // multigpu_device_next_proposed_w_[gpu_id].resize(arr_size);
        // multigpu_device_partner_rank_[gpu_id].resize(arr_size);

        thrust::fill(thrust::cuda::par.on(gpu_streams_[gpu_id].cuda_stream()),
                     multigpu_device_next_proposed_w_[gpu_id].begin(),
                     multigpu_device_next_proposed_w_[gpu_id].end(), 0);

        thrust::fill(thrust::cuda::par.on(gpu_streams_[gpu_id].cuda_stream()),
                     multigpu_device_partner_rank_[gpu_id].begin(),
                     multigpu_device_partner_rank_[gpu_id].end(), n_);

        // 2-d array init
        // int element_offset = arr_offset * n_;
        // int element_size = arr_size * n_;
        size_t element_offset = IDX_MUL(arr_offset, n_);
        size_t element_size = IDX_MUL(arr_size, n_);

        LOG(INFO) << "GPU: " << gpu_id
            << ", element_offset: " << element_offset << ", element_size: " << element_size;

        multigpu_device_pref_lists_m_.emplace_back(element_size);
        multigpu_device_pref_lists_w_.emplace_back(element_size);
        multigpu_device_rank_mtx_w_.emplace_back(element_size);
        multigpu_device_prnodes_m_.emplace_back(element_size);

        CHECK(multigpu_device_pref_lists_m_.size() > gpu_id);
        CHECK(multigpu_device_pref_lists_w_.size() > gpu_id);
        CHECK(multigpu_device_rank_mtx_w_.size() > gpu_id);
        CHECK(multigpu_device_prnodes_m_.size() > gpu_id);

        // multigpu_device_pref_lists_m_[gpu_id].resize(element_size);
        // multigpu_device_pref_lists_w_[gpu_id].resize(element_size);
        // multigpu_device_rank_mtx_w_[gpu_id].resize(element_size);
        // multigpu_device_prnodes_m_[gpu_id].resize(element_size);

        SafeCudaMemcpyAsync(multigpu_device_pref_lists_m_[gpu_id],
                            RawPtr(smp_.flatten_pref_lists_m_vec) + element_offset,
                            element_size,
                            cudaMemcpyHostToDevice,
                            gpu_streams_[gpu_id].cuda_stream(),
                            "pref_lists_m");

        SafeCudaMemcpyAsync(multigpu_device_pref_lists_w_[gpu_id],
                            RawPtr(smp_.flatten_pref_lists_w_vec) + element_offset,
                            element_size,
                            cudaMemcpyHostToDevice,
                            gpu_streams_[gpu_id].cuda_stream(),
                            "pref_lists_w");

        LOG(INFO) << "GPU: " << gpu_id << " cudaMemcpyAsync launched.";

        // After each GPU allocation, re-check GPU 0 memory explicitly
        cudaSetDevice(0);
        size_t free_bytes_gpu0_now, total_bytes_gpu0_now;
        cudaMemGetInfo(&free_bytes_gpu0_now, &total_bytes_gpu0_now);
        LOG(INFO) << "[After GPU " << gpu_id << " alloc] GPU 0 memory: "
            << free_bytes_gpu0_now / (1024 * 1024) << " MB free of "
            << total_bytes_gpu0_now / (1024 * 1024) << " MB.";
      }

      SYNC_ALL_STREAMS(gpu_streams_)
      LOG(INFO) << "AsyncCopy: After syncing all streams.";

      // Log remaining GPU memory
      FOR_EACH_GPU(num_gpus_, gpu_id) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        LOG(INFO) << "GPU " << gpu_id << " has " << free_bytes / (1024 * 1024)
            << " MB free out of " << total_bytes / (1024 * 1024) << " MB.";
        cudaDeviceSynchronize();
      }
    }
  }

  void SmpEngineMultiGpu::InitMatriceAsync() {
    LOG(INFO) << "Starting AsyncInitMatriceOnGpus";
    // Launch RankMatrix initialization kernels asynchronously on each GPU

    {
      PROFILE_SCOPE("InitRankMatrix");

      FOR_EACH_GPU(num_gpus_, gpu_id) {
        int local_size = PARTITION_SIZE(n_, num_gpus_, gpu_id);
        int element_size = local_size * n_;

        LaunchKernelForEach(gpu_streams_[gpu_id],
                            element_size,
                            InitRankMatrixKernelTasklet{
                              n_,
                              ConstArrayView<int>(multigpu_device_pref_lists_w_[gpu_id]),
                              ArrayView<int>(multigpu_device_rank_mtx_w_[gpu_id])
                            });
      }

      SYNC_ALL_STREAMS(gpu_streams_)
      LOG(INFO) << "RankMatrix kernels completed successfully";
    } {
      PROFILE_SCOPE("InitPRMatrix");

      MultiGpuArrayViewManager<int> multi_gpu_array_view_manager(multigpu_device_rank_mtx_w_);

      FOR_EACH_GPU(num_gpus_, gpu_id) {
        int local_size = PARTITION_SIZE(n_, num_gpus_, gpu_id);
        int element_size = local_size * n_;

        LaunchKernelForEach(gpu_streams_[gpu_id],
                            element_size,
                            InitPRMatrixMultiGpuKernelTasklet{
                              n_,
                              gpu_id,
                              num_gpus_,
                              ConstArrayView<int>(multigpu_device_pref_lists_m_[gpu_id]),
                              // device_rank_mtx_ptrs[gpu_id],
                              multi_gpu_array_view_manager.GetMultiGpuArrayView(gpu_id),
                              ArrayView<PRNode>(multigpu_device_prnodes_m_[gpu_id])
                            }
        );
      }
      SYNC_ALL_STREAMS(gpu_streams_);

      FOR_EACH_GPU(num_gpus_, gpu_id) {
        cudaDeviceSynchronize();
      }

      LOG(INFO) << "PRMatrix kernels completed successfully";
      LOG(INFO) << "Completed AsyncInitMatriceOnGpus";
    }
  }

  auto SmpEngineMultiGpu::RetrievePRMatrixForTest() -> Vector<PRNode> {
    Vector<PRNode> prmatrix(n_ * n_);
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      int local_offset = PARTITION_START(n_, num_gpus_, gpu_id);

      // int element_offset = local_offset * n_;
      size_t element_offset = IDX_MUL(local_offset, n_);

      thrust::copy(multigpu_device_prnodes_m_[gpu_id].begin(),
                   multigpu_device_prnodes_m_[gpu_id].end(),
                   prmatrix.begin() + element_offset);
    }
    return prmatrix;
  }

  auto SmpEngineMultiGpu::RetrieveRankMatrixForTest() -> Vector<int> {
    Vector<int> merged(n_ * n_);
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      int local_offset = PARTITION_START(n_, num_gpus_, gpu_id);

      // int element_offset = local_offset * n_;
      size_t element_offset = IDX_MUL(local_offset, n_);

      thrust::copy(multigpu_device_rank_mtx_w_[gpu_id].begin(),
                   multigpu_device_rank_mtx_w_[gpu_id].end(),
                   merged.begin() + element_offset);
    }
    return merged;
  }


  /* ------------------------------------- CoreProc ------------------------------------------------ */
  void SmpEngineMultiGpu::CoreProc() {
    LOG(INFO) << "SmpEngineMultiGpu::CoreProc Starts";
    std::thread thread_cpu(&SmpEngineMultiGpu::DoWorkOnCpu, this, tflag_);
    std::thread thread_gpu(&SmpEngineMultiGpu::DoWorkOnGpu, this, tflag_);

    TFlagValue mode = TFlagValue::NotFinished;
    while (mode == TFlagValue::NotFinished) {
      SLEEP_MILLISECONDS(1);
      mode = tflag_->GetFlag();
    }

    if (mode == TFlagValue::FinishedByCpu) {
      LOG(INFO) << "Computation finished by CPU.";
      thread_cpu.join();
      thread_gpu.detach();
      // thread_gpu.join();
    } else if (mode == TFlagValue::FinishedByGpu) {
      LOG(INFO) << "Computation finished by GPU.";
      thread_gpu.join();
      thread_cpu.detach();
    }
    LOG(INFO) << "SmpEngineMultiGpu::CoreProc Ends";
  }

  void SmpEngineMultiGpu::DoWorkOnGpu(SPtr<AtomicTFlag> tflag) {
    MultiGpuArrayViewManager<PRNode> multigpu_prnodes_m_avmgr(this->multigpu_device_prnodes_m_);
    MultiGpuArrayViewManager<int> multigpu_pref_lists_w_avmgr(this->multigpu_device_pref_lists_w_);
    MultiGpuArrayViewManager<int> multigpu_next_proposed_w_avmgr(this->multigpu_device_next_proposed_w_);
    MultiGpuArrayViewManager<int> multigpu_partner_rank_avmgr(this->multigpu_device_partner_rank_); {
      PROFILE_SCOPE("LAKernel");
      FOR_EACH_GPU(num_gpus_, gpu_id) {
        LaunchKernelForEach(gpu_streams_[gpu_id],
                            PARTITION_SIZE(n_, num_gpus_, gpu_id),
                            LAMultiGPUKernelTasklet{
                              n_,
                              gpu_id,
                              num_gpus_,
                              multigpu_prnodes_m_avmgr.GetMultiGpuArrayView(gpu_id),
                              multigpu_pref_lists_w_avmgr.GetMultiGpuArrayView(gpu_id),
                              multigpu_partner_rank_avmgr.GetMultiGpuArrayView(gpu_id),
                              multigpu_next_proposed_w_avmgr.GetMultiGpuArrayView(gpu_id)
                            });
      }
      SYNC_ALL_STREAMS(gpu_streams_);
    }

    if (tflag->TrySetTFlag(TFlagValue::FinishedByGpu)) {
      FOR_EACH_GPU(num_gpus_, gpu_id) {
        int arr_offset = PARTITION_START(n_, num_gpus_, gpu_id);
        thrust::copy(multigpu_device_partner_rank_[gpu_id].begin(),
                     multigpu_device_partner_rank_[gpu_id].end(),
                     final_partner_ranks_.begin() + arr_offset);
      }
      LOG(INFO) << "SmpEngineMultiGpu::BambooKernel (GPU) won the contention.";
    } else {
      LOG(INFO) << "SmpEngineMultiGpu::BambooKernel (GPU) fails the contention.";
    }
  }


  void SmpEngineMultiGpu::DoWorkOnCpu(SPtr<AtomicTFlag> tflag) {
    PROFILE_SCOPE("Total Cpu Time");
    LOG(INFO) << "CPU Monitoring procedure starts.";
    int unmatched_id, unmatched_num;

    uint64_t start_ns = getNanoSecond();

    HVector<int> host_partner_rank;
    host_partner_rank.resize(n_);

    // this->MonitorProceture(unmatched_id, unmatched_num, final_partner_ranks_, monitor_streams_);
    this->MonitorProceture(unmatched_id, unmatched_num, host_partner_rank, monitor_streams_);

    if (unmatched_num == 1) {
      // CUDA_CHECK(cudaMallocHost(&host_prnodes_m_ptr_, n_ * n_ * sizeof(PRNode)));
      CUDA_CHECK(cudaMallocHost(&host_prnodes_m_ptr_, SIZE_MUL(n_, n_) * sizeof(PRNode)));
      // host_prnodes_m_.resize(SIZE_MUL(n_, n_));
      // host_prnodes_m_ptr_ = host_prnodes_m_.data();

      auto prnodes_h2d_start = getNanoSecond();

      LOG(INFO) << "Monitor detected exactly one unmatched man. Proceeding to LAProcedure.";
      this->PRNodesD2H(host_prnodes_m_ptr_, monitor_streams_);

      auto prnodes_h2d_end = getNanoSecond();
      std::cout << "[MultiGPU:DoWorkOnCpu] total D2D took took " << (prnodes_h2d_end - prnodes_h2d_start) / 1e6 <<
          " ms.";

      // auto host_next_proposed_w_view = ArrayView<int>(host_next_proposed_w);
      // auto host_partner_rank_view = ArrayView<int>(host_partner_ranks);
      // auto pref_lists_w_view = ArrayView<int>(smp_.flatten_pref_lists_w_);
      // ArrayView<PRNode> host_prnodes_m_view(host_prnodes_m_ptr_, n_ * n_);

      auto start_la = getNanoSecond();
      LAProcedure(unmatched_id,
                  // ArrayView<PRNode>(this->host_prnodes_m_ptr_, n_ * n_),
                  ArrayView<PRNode>(this->host_prnodes_m_ptr_, SIZE_MUL(n_, n_)),
                  ArrayView<int>(host_partner_rank),
                  ArrayView<int>(this->host_next_proposed_w_),
                  ArrayView<int>(this->smp_.flatten_pref_lists_w_vec));
      auto stop_la = getNanoSecond();
      std::cout << "[MultiGPU::LAProcedure] takes time" << (stop_la - start_la) / 1e6 << std::endl;

      if (tflag->TrySetTFlag(TFlagValue::FinishedByCpu)) {
        final_partner_ranks_ = host_partner_rank;
        LOG(INFO) << "SmpEngineMultiGpu::CPU won the contention.";
        std::cout << "SmpEngineMultiGpu::CPU won the contention." << std::endl;
      } else {
        LOG(INFO) << "SmpEngineMultiGpu::CPU fails the contention.";
        std::cout << "SmpEngineMultiGpu::CPU fails the contention." << std::endl;
      }
    }
    uint64_t end_ns = getNanoSecond();
    std::cout << "[MultiGPU:DoWorkOnCpu] total execution took " << (end_ns - start_ns) / 1e6 << " ms.";
  }

  void SmpEngineMultiGpu::MonitorProceture(int &unmatched_id,
                                           int &unmatched_num,
                                           HVector<int> &host_partner_rank,
                                           const Vector<CudaStream> &monitor_streams) {
    auto pref_lists_w_view = ArrayView<int>(smp_.flatten_pref_lists_w_vec);
    auto host_partner_rank_view = ArrayView<int>(host_partner_rank);

    // const int total = n_ * (n_ - 1) / 2;
    const size_t total = IDX_MUL(n_, (n_ - 1)) / 2;

    size_t iteration = 0;
    do {
      SLEEP_MILLISECONDS(1000);
      PartnerRanksD2H(host_partner_rank, monitor_streams);

      unmatched_id = total;
      unmatched_num = 0;

      for (int w = 0; w < n_; ++w) {
        int m_rank = host_partner_rank_view[w];
        if (m_rank == n_) {
          unmatched_num++;
        } else {
          // unmatched_id -= pref_lists_w_view[w * n_ + m_rank];
          unmatched_id -= pref_lists_w_view[IDX_MUL_ADD(w, n_, m_rank)];
        }
      }

      LOG(INFO) << "[MonitorProcedure] Iteration " << iteration << ": unmatched men = " << unmatched_num;
      iteration++;
    } while (unmatched_num > 1);
  }

  void SmpEngineMultiGpu::LAProcedure(int m,
                                      ArrayView<PRNode> host_prnodes_m_view,
                                      ArrayView<int> host_partner_ranks_view,
                                      ArrayView<int> host_next_proposed_w_view,
                                      ArrayView<int> pref_lists_w_view) {
    // LOG(INFO) << "[LAProcedure] starts on CPU with unmatched man: " << m;
    std::cout << "[LAProcedure] starts on CPU with unmatched man: " << m << std::endl;

    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node;
    bool paired = false;
    size_t iterations = 0;
    while (!paired) {
      iterations += 1;
      // if (iterations % 2000 == 0) {
      //   LOG(INFO) << "[LAProcedure] iteration: " << iterations << ", current man: " << m_idx << ", current rank: " << w_rank;
      //   std::cout << "[LAProcedure] iteration: " << iterations << ", current man: " << m_idx << ", current rank: " << w_rank << std::endl;
      // }

      // temp_node = host_prnodes_m_view[m_idx * n_ + w_rank];
      temp_node = host_prnodes_m_view[IDX_MUL_ADD(m_idx, n_, w_rank)];

      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      p_rank = host_partner_ranks_view[w_idx];
      if (p_rank == n_) {
        host_next_proposed_w_view[m_idx] = w_rank;
        host_partner_ranks_view[w_idx] = m_rank;
        paired = true;
      } else if (p_rank > m_rank) {
        host_next_proposed_w_view[m_idx] = w_rank;
        host_partner_ranks_view[w_idx] = m_rank;

        // m_idx = pref_lists_w_view[w_idx * n_ + p_rank];
        m_idx = pref_lists_w_view[IDX_MUL_ADD(w_idx, n_, p_rank)];

        w_rank = host_next_proposed_w_view[m_idx];
      } else {
        w_rank++;
      }
    }
    // LOG(INFO) << "[LAProcedure] successfully completed after iterations: " << iterations;
    std::cout << "[LAProcedure] successfully completed after iterations: " << iterations << std::endl;
  }


  void SmpEngineMultiGpu::PartnerRanksD2H(HVector<int> &host_partner_rank,
                                          const Vector<CudaStream> &monitor_streams) {
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      int arr_offset = PARTITION_START(n_, num_gpus_, gpu_id);
      int arr_size = PARTITION_SIZE(n_, num_gpus_, gpu_id);

      CUDA_CHECK(cudaMemcpyAsync(RawPtr(host_partner_rank) + arr_offset,
        RawPtr(multigpu_device_partner_rank_[gpu_id]),
        arr_size * sizeof(int),
        cudaMemcpyDeviceToHost,
        monitor_streams[gpu_id].cuda_stream()));
    }

    SYNC_ALL_STREAMS(monitor_streams);
  }

  void SmpEngineMultiGpu::PRNodesD2H(PRNode *host_prnodes_m_ptr,
                                     const Vector<CudaStream> &monitor_streams) {
    LOG(INFO) << "[PRNodesD2H] Starting host-device copy for PRNodes.";
    FOR_EACH_GPU(num_gpus_, gpu_id) {
      int arr_offset = PARTITION_START(n_, num_gpus_, gpu_id);
      int arr_size = PARTITION_SIZE(n_, num_gpus_, gpu_id);

      // int element_offset = arr_offset * n_;
      // int element_size = arr_size * n_;
      size_t element_offset = IDX_MUL(arr_offset, n_);
      size_t element_size = IDX_MUL(arr_size, n_);

      CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_ptr + element_offset,
        RawPtr(multigpu_device_prnodes_m_[gpu_id]),
        element_size * sizeof(PRNode),
        cudaMemcpyDeviceToHost,
        monitor_streams[gpu_id].cuda_stream()));
    }


    LOG(INFO) << "[PRNodesD2H] All memcpyAsync issued. Waiting for sync...";
    SYNC_ALL_STREAMS(monitor_streams);
    LOG(INFO) << "[PRNodesD2H] Host-device copy for PRNodes completed.";
  }

  /* ------------------------------------- PostProc ------------------------------------------------ */
  void SmpEngineMultiGpu::PostProc() {
    // Convert rank back to actual matching ID
    for (int w = 0; w < n_; w++) {
      // int m = smp_.flatten_pref_lists_w_[w * n_ + host_partner_ranks_[w]];
      int m = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w, n_, final_partner_ranks_[w])];
      stable_matching_[m] = w;
    }
  }

  /* ------------------------------------- Return Stable Marriage ------------------------------------------------ */
  auto SmpEngineMultiGpu::GetStableMarriage() const -> std::vector<int> {
    std::vector<int> result(n_);
    for (int i = 0; i < n_; i++) {
      result[i] = stable_matching_[i];
    }
    return result;
  }
}
