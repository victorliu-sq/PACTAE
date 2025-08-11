#ifndef KERNEL_H
#define KERNEL_H

#pragma once
#include <cuda_runtime.h>
#include <utils/array_view.h>
#include <utils/array_view_multigpu.h>

struct InitRankMatrixKernelTasklet {
  int n;
  ConstArrayView<int> pref_lists;
  ArrayView<int> rank_mtx;

  __device__ void operator()(size_t tid) const {
    int m_idx, w_idx, m_rank;
    w_idx = tid / n;
    m_rank = tid % n;
    // m_idx = pref_lists[w_idx * n + m_rank];
    m_idx = pref_lists[IDX_MUL_ADD(w_idx, n, m_rank)];
    // rank_mtx[w_idx * n + m_idx] = m_rank;
    rank_mtx[IDX_MUL_ADD(w_idx, n, m_idx)] = m_rank;
  }
};

struct InitPRMatrixKernelTasklet {
  int n;
  ConstArrayView<int> rank_mtx_w;
  ConstArrayView<int> pref_lists_m;
  ArrayView<PRNode> prnodes_m;

  __device__ void operator()(size_t tid) const {
    // PRNode node;
    int m_idx = tid / n;
    int w_rank = tid % n;

    // int w_idx = pref_lists_m[m_idx * n + w_rank];
    int w_idx = pref_lists_m[IDX_MUL_ADD(m_idx, n, w_rank)];
    // int m_rank = rank_mtx_w[w_idx * n + m_idx];
    int m_rank = rank_mtx_w[IDX_MUL_ADD(w_idx, n, m_idx)];

    // node.idx_ = w_idx;
    // node.rank_ = m_rank;
    // prnodes_m[m_idx * n + w_rank] = {w_idx, m_rank};
    prnodes_m[IDX_MUL_ADD(m_idx, n, w_rank)] = {w_idx, m_rank};
  }
};


struct LAKernelTasklet {
  int n;
  ConstArrayView<PRNode> prnodes_m;
  ConstArrayView<int> pref_lists_w;
  ArrayView<int> partner_rank;
  ArrayView<int> next_proposed_w;

  __device__ void operator()(size_t tid) {
    int mi, mi_rank, w_idx, w_rank, mj_rank;
    mi = tid;
    w_rank = 0;
    PRNode node;
    bool paired = false;
    while (!paired) {
      // node = prnodes_m[mi * n + w_rank];
      node = prnodes_m[IDX_MUL_ADD(mi, n, w_rank)];
      w_idx = node.idx_;
      mi_rank = node.rank_;
      w_rank += 1;
      // if (partner_rank[w_idx] < mi_rank) {
      //   continue;
      // }
      mj_rank = atomicMin(&partner_rank[w_idx], mi_rank);
      if (mj_rank > mi_rank) {
        next_proposed_w[mi] = w_rank;
        if (mj_rank == n) {
          paired = true;
        } else {
          // mi = pref_lists_w[w_idx * n + mj_rank];
          mi = pref_lists_w[IDX_MUL_ADD(w_idx, n, mj_rank)];
          w_rank = next_proposed_w[mi];
        }
      }
    }
  }
};

/* -------------- same as the single gpu version ------------------
struct InitRankMatrixMultiGpuKernelTasklet {
  int n;
  int gpu_id;
  int num_gpus;
  ConstArrayView<int> pref_lists;
  ArrayView<int> rank_mtx;

  __device__ void operator()(size_t tid) const {
    int m_idx, local_w_idx, m_rank;
    local_w_idx = tid / n;
    m_rank = tid % n;
    m_idx = pref_lists[local_w_idx * n + m_rank];
    rank_mtx[local_w_idx * n + m_idx] = m_rank;
  }
};
*/

__device__ inline int GetOwnerGpuId(int global_idx, int total_rows, int num_gpus) {
  int rows_per_gpu = total_rows / num_gpus;
  int extra_rows = total_rows % num_gpus;
  int threshold = (rows_per_gpu + 1) * extra_rows;

  if (global_idx < threshold) {
    // GPUs that have an extra row
    return global_idx / (rows_per_gpu + 1);
  } else {
    // GPUs with standard number of rows
    return extra_rows + (global_idx - threshold) / rows_per_gpu;
  }
}

struct InitPRMatrixMultiGpuKernelTasklet {
  int n;
  int gpu_id;
  int num_gpus;
  ConstArrayView<int> pref_lists_m;
  MultiGpuArrayView<int> multi_gpu_rank_mtx_w;
  ArrayView<PRNode> prnodes_m;

  __device__ void operator()(size_t tid) const {
    int local_m_idx = tid / n;
    int w_rank = tid % n;
    // int global_w_idx = pref_lists_m[local_m_idx * n + w_rank];
    int global_w_idx = pref_lists_m[IDX_MUL_ADD(local_m_idx, n, w_rank)];

    int partition_start_m_idx = PARTITION_START(n, num_gpus, gpu_id);
    int global_m_idx = local_m_idx + partition_start_m_idx;

    int owner_gpu = GetOwnerGpuId(global_w_idx, n, num_gpus);
    int partition_start_owner_w_idx = PARTITION_START(n, num_gpus, owner_gpu);
    int owner_local_w_idx = global_w_idx - partition_start_owner_w_idx;

    // int m_rank = multi_gpu_rank_mtx_w(owner_gpu, owner_local_w_idx * n + global_m_idx);
    int m_rank = multi_gpu_rank_mtx_w(owner_gpu, IDX_MUL_ADD(owner_local_w_idx, n, global_m_idx));
    // prnodes_m[local_m_idx * n + w_rank] = {global_w_idx, m_rank};
    prnodes_m[IDX_MUL_ADD(local_m_idx, n, w_rank)] = {global_w_idx, m_rank};
  }
};

struct LAMultiGPUKernelTasklet {
  int n;
  int gpu_id;
  int num_gpus;
  MultiGpuArrayView<PRNode> prnodes_m;
  MultiGpuArrayView<int> pref_lists_w;
  MultiGpuArrayView<int> partner_rank;
  MultiGpuArrayView<int> next_proposed_w;

  __device__ void operator()(size_t tid) {
    int partition_start_m_idx = PARTITION_START(n, num_gpus, gpu_id);
    int local_m_idx = tid;
    int global_m_idx = local_m_idx + partition_start_m_idx;

    int w_rank = 0;
    bool paired = false;

    int owner_id_global_m_idx = gpu_id;
    int owner_local_m_idx = local_m_idx;

    while (!paired) {
      // PRNode node = prnodes_m(owner_id_global_m_idx, owner_local_m_idx * n + w_rank);
      PRNode node = prnodes_m(owner_id_global_m_idx, IDX_MUL_ADD(owner_local_m_idx, n, w_rank));
      int global_w_idx = node.idx_;
      int mi_rank = node.rank_;
      w_rank++;

      int owner_id_global_w_idx = GetOwnerGpuId(global_w_idx, n, num_gpus);
      int owner_local_w_idx = global_w_idx - PARTITION_START(n, num_gpus, owner_id_global_w_idx);
      int mj_rank = atomicMin(&partner_rank(owner_id_global_w_idx, owner_local_w_idx), mi_rank);

      if (mj_rank > mi_rank) {
        next_proposed_w(owner_id_global_m_idx, owner_local_m_idx) = w_rank;

        if (mj_rank == n) {
          paired = true;
        } else {
          // global_m_idx = pref_lists_w(owner_id_global_w_idx, owner_local_w_idx * n + mj_rank);
          global_m_idx = pref_lists_w(owner_id_global_w_idx, IDX_MUL_ADD(owner_local_w_idx, n, mj_rank));

          owner_id_global_m_idx = GetOwnerGpuId(global_m_idx, n, num_gpus);
          owner_local_m_idx = global_m_idx - PARTITION_START(n, num_gpus, owner_id_global_m_idx);
          w_rank = next_proposed_w(owner_id_global_m_idx, owner_local_m_idx);
        }
      }
    }
  }
};


#endif //KERNEL_H
