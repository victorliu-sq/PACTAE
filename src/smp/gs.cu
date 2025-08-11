#include <cstdint>
#include <iostream>
#include <smp/gs.cuh>
#include <smp/smp.h>
#include <thread>
#include <vector>
#include <smp/kernel.h>
#include <utils/launcher.h>
#include <utils/stopwatch.h>
#include <utils/utils.h>

namespace bamboosmp {
  GS::GS(const SmpObj &smp, int thread_limit, int size)
    : smp_(smp), n_(size), num_threads_per_block_(thread_limit) {
    // InitGpu();
    // InitCpu();
  }

  // __attribute__((optimize("O0")))
  void InitRankMatrixSeq(int *host_rank_mtx_w_, const SmpObj &smp_, int n_) {
    for (int w_idx = 0; w_idx < n_; ++w_idx) {
      for (int m_rank = 0; m_rank < n_; ++m_rank) {
        int m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
        host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
      }
    }
  }

  void GS::InitGpu() {
    float time_copy_to_gpu, time_gpu_execution, time_copy_to_cpu,
        time_total_preprocessing;

    StopWatch sw_alloc;
    size_t mtx_size = SIZE_MUL(n_, n_);
    // host_rank_mtx_w_ = new int[mtx_size];
    CUDA_CHECK(cudaMallocHost(&host_rank_mtx_w_, mtx_size * sizeof(int),
      cudaHostAllocDefault));

    // husband_rank_ = new int[n_];
    CUDA_CHECK(
      cudaMallocHost(&husband_rank_, n_ * sizeof(int), cudaHostAllocDefault));

    next_proposed_w_ = new int[n_];

    atomic_partner_rank_ = new std::atomic<int>[n_];

    // CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_m_, mtx_size *
    // sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_w_, mtx_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&device_rank_mtx_w_, mtx_size * sizeof(int)));

    CUDA_CHECK(cudaDeviceSynchronize());
    sw_alloc.Stop();
    std::cout << "Time to allocate gpu resources is : " << sw_alloc.GetEclapsedMs() << " ms." << std::endl;

    cudaError_t err;
    uint64_t start_memcpy, end_memcpy;
    start_memcpy = getNanoSecond();
    CUDA_CHECK(cudaMemcpy(device_pref_lists_w_, RawPtr(smp_.flatten_pref_lists_w_vec),
      mtx_size * sizeof(int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    end_memcpy = getNanoSecond();
    time_copy_to_gpu = (end_memcpy - start_memcpy) * 1.0 / 1e6;
    std::cout << "GS Copy PrefLists into GPU spends " << time_copy_to_gpu
        << " ms " << std::endl;

    // p >= n^2
    // int threadsPerBlock = num_threads_per_block_;

    // int init_num_blocks = (mtx_size + threadsPerBlock - 1) / threadsPerBlock;
    // printf("Initialization: Launch %d blocks\n", init_num_blocks);

    auto start_init_nodes = getNanoSecond();

    // InitRankMatrix<<<init_num_blocks, num_threads_per_block_>>>(
    //   n_, device_rank_mtx_w_, device_pref_lists_w_);

    CudaStream stream;
    LaunchKernelForEach(stream,
                        mtx_size,
                        InitRankMatrixKernelTasklet{
                          n_,
                          ConstArrayView<int>(device_pref_lists_w_, mtx_size),
                          ArrayView<int>(device_rank_mtx_w_, mtx_size),
                        });

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "GSInit Launch err: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    auto end_init_nodes = getNanoSecond();

    time_gpu_execution = (end_init_nodes - start_init_nodes) * 1.0 / 1e6;
    std::cout << "GS Init RankMatrixW  in parallel on GPU spends "
        << time_gpu_execution << " ms" << std::endl;

    // Copy back to CPU
    start_memcpy = getNanoSecond();
    CUDA_CHECK(cudaMemcpy(host_rank_mtx_w_, device_rank_mtx_w_,
      mtx_size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize(); // Ensure prefetching is complete
    end_memcpy = getNanoSecond();
    time_copy_to_cpu = (end_memcpy - start_memcpy) * 1.0 / 1e6;
    std::cout << "GS Copy back RankMatrixW to CPU spends " << time_copy_to_cpu
        << " ms " << std::endl;

    time_total_preprocessing =
        time_copy_to_gpu + time_copy_to_cpu + time_gpu_execution;

    std::cout << "Total GS preprocessing time is " << time_total_preprocessing
        << " ms " << std::endl;

    auto start_time = getNanoSecond();

    // std::cout << "Init husband_rank starts" << std::endl;
    for (int i = 0; i < n_; ++i) {
      free_men_queue_.push(i);
      husband_rank_[i] = n_;

      // std::cout << "husband_rank[" << i << "] is " << husband_rank_[i]
      //           << std::endl;
      next_proposed_w_[i] = 0;
      atomic_partner_rank_[i].store(n_);
    }
    // std::cout << "Init husband_rank is done" << std::endl;

    auto end_time = getNanoSecond();
    auto time = (end_time - start_time) * 1.0 / 1e6;
    std::cout << "GS init other data structures require " << time << " ms "
        << std::endl;

    max_num_threads_ =
        std::thread::hardware_concurrency(); // Get the number of available
    // hardware threads
    // num_threads_ = 12;

    if (max_num_threads_ > n_) {
      max_num_threads_ = n_;
    }

    // std::cout << "Total number of threads is " << num_threads_ << std::endl;

    // initialize queues for parallel GS
    free_men_queues_ = initialize_multiple_queues(max_num_threads_, n_);
  }

  void GS::InitCpuSeq() {
    size_t mtx_size = SIZE_MUL(n_, n_);
    host_rank_mtx_w_ = new int[mtx_size];

    max_num_threads_ = std::min(std::thread::hardware_concurrency(), 96u);

    StopWatch sw_init_rank_mtx;
    InitRankMatrixSeq(host_rank_mtx_w_, smp_, n_);
    // for (int w_idx = 0; w_idx < n_; ++w_idx) {
    //   for (int m_rank = 0; m_rank < n_; ++m_rank) {
    //     int m_idx = smp_.flatten_pref_lists_w_[IDX_MUL_ADD(w_idx, n_, m_rank)];
    //     host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
    //   }
    // }
    sw_init_rank_mtx.Stop();
    auto time_init_rank_mtx = sw_init_rank_mtx.GetEclapsedMs();
    std::cout << "GS Init Rank Mtx Seq spends time " << time_init_rank_mtx << " ms." << std::endl;

    // Initialize additional data structures
    husband_rank_ = new int[n_];
    next_proposed_w_ = new int[n_];
    atomic_partner_rank_ = new std::atomic<int>[n_];

    for (int i = 0; i < n_; ++i) {
      free_men_queue_.push(i);
      husband_rank_[i] = n_;
      next_proposed_w_[i] = 0;
      atomic_partner_rank_[i].store(n_);
    }

    free_men_queues_ = initialize_multiple_queues(max_num_threads_, n_);
  }


  void GS::InitCpuPar() {
    size_t mtx_size = SIZE_MUL(n_, n_);
    host_rank_mtx_w_ = new int[mtx_size];

    auto start_time = getNanoSecond();

    max_num_threads_ = std::min(std::thread::hardware_concurrency(), 96u);
    const int *pref_list_w_ptr = RawPtr(smp_.flatten_pref_lists_w_vec);

    // auto thread_task = [this, pref_list_w_ptr](int w_idx) {
    //   for (int m_rank = 0; m_rank < n_; ++m_rank) {
    //     int m_idx = pref_list_w_ptr[IDX_MUL_ADD(w_idx, n_, m_rank)];
    //     host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
    //   }
    // };

    // for (int w_idx = 0; w_idx < n_; ++w_idx) {
    //   if (threads.size() >= max_num_threads_) {
    //     // Wait for current batch to finish
    //     for (auto &th: threads) {
    //       th.join();
    //     }
    //     threads.clear();
    //   }
    //
    //   threads.emplace_back(thread_task, w_idx);
    // }

    const int num_blocks = max_num_threads_;
    std::vector<std::thread> threads(num_blocks);

    auto thread_task = [&](int start_row, int end_row) {
      for (int w_idx = start_row; w_idx < end_row; ++w_idx) {
        for (int m_rank = 0; m_rank < n_; ++m_rank) {
          int m_idx = smp_.flatten_pref_lists_w_vec[IDX_MUL_ADD(w_idx, n_, m_rank)];
          host_rank_mtx_w_[IDX_MUL_ADD(w_idx, n_, m_idx)] = m_rank;
        }
      }
    };

    int rows_per_block = n_ / num_blocks;

    for (int t = 0; t < num_blocks; ++t) {
      int start_row = t * rows_per_block;
      int end_row = (t == num_blocks - 1) ? n_ : start_row + rows_per_block;
      threads[t] = std::thread(thread_task, start_row, end_row);
    }

    // Join any remaining threads
    for (auto &th: threads) {
      th.join();
    }

    auto end_time = getNanoSecond();
    double time_taken = (end_time - start_time) / 1e6;

    std::cout << "CPU parallel initialization of RankMatrixW (1 thread per woman) took "
        << time_taken << " ms" << std::endl;

    // Initialize additional data structures
    husband_rank_ = new int[n_];
    next_proposed_w_ = new int[n_];
    atomic_partner_rank_ = new std::atomic<int>[n_];

    for (int i = 0; i < n_; ++i) {
      free_men_queue_.push(i);
      husband_rank_[i] = n_;
      next_proposed_w_[i] = 0;
      atomic_partner_rank_[i].store(n_);
    }

    free_men_queues_ = initialize_multiple_queues(max_num_threads_, n_);
  }


  // Function to initialize a single queue
  void GS::initialize_queue(std::queue<int> &q, int start, int end) {
    // printf("man %d ~ %d have been pushed into queue\n", start, end - 1);
    for (int i = start; i < end; ++i) {
      q.push(i);
    }
  }

  // Function to initialize multiple queues in parallel
  auto GS::initialize_multiple_queues(int num_threads, int n)
    -> std::vector<std::queue<int> > {
    std::vector<std::queue<int> > queues(num_threads);
    std::vector<std::thread> threads;

    int avg_num_men = n / num_threads;

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
      if (threads.size() >= num_threads) {
        for (auto &th: threads) {
          th.join();
        }
        threads.clear();
      }

      int start = i * avg_num_men;
      int end = (i != num_threads - 1) ? (i + 1) * avg_num_men : n;

      threads.emplace_back(&GS::initialize_queue, this,
                           std::ref(queues[i]), start, end);
    }


    // Join threads
    for (auto &th: threads) {
      th.join();
    }

    return queues;
  }

  GS::~GS() {
    /*
    // delete[] husband_rank_;
    delete[] next_proposed_w_;
    CUDA_CHECK(cudaFree(device_pref_lists_w_));
    // CUDA_CHECK(cudaFree(device_pref_lists_m_));
    CUDA_CHECK(cudaFree(device_rank_mtx_w_));

    // delete[] host_rank_mtx_w_;
    CUDA_CHECK(cudaFreeHost(host_rank_mtx_w_));
    // cudaDeviceReset()
    // cudaDeviceSynchronize();
    */

    delete[] host_rank_mtx_w_;
    delete[] husband_rank_;
    delete[] next_proposed_w_;
    delete[] atomic_partner_rank_;
  }
}
