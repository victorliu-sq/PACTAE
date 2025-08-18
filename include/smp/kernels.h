#ifndef KERNELS_H
#define KERNELS_H
#include "utils/types.h"

// ====================== Table 1 ==============================
__global__ void Mw3T1CoreKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

// ====================== Figure 7 ==============================
__global__ void La2F7CoreKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
);

__global__ void Mw3F7CoreKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

// ================================= Figrue8 Mw3(CAS) ==================================
__global__ void Mw3F8CASTopKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

__global__ void Mw3F8CASBottomKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

// ================================= Figrue8 La2(CAS) ==================================
__global__ void La2F8CASTopKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
);

__global__ void La2F8CASBottomKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
);

// ================================= Figrue8 Mw4(MIN) ==================================
__global__ void Mw4F8BottomKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

__global__ void Mw4F8TopLeftKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

__global__ void Mw4F8TopRightKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
);

// ================================= Figrue8 La3(MIN) ==================================
__global__ void La3MinF8BottomKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
);

__global__ void La3MinF8TopLeftKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
);

__global__ void La3MinF8TopRightKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
);

// ====================== Figure 9 ==============================
__global__ void La3F9CoreKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
);

__global__ void F9RankMatrixInitKernel(
  int n,
  const int * pref_list_w_dview,
  int * rank_matrix_dview
);

__global__ void F9PRMatrixInitKernel(
  int n,
  const int * pref_list_m_dview,
  const int * rank_matrix_dview,
  PRNode* pr_mtx
);


#endif //KERNELS_H
