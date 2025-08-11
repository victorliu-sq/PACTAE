#include "smp/kernels.h"
#include "utils/utils.h"

// ================================= Table 1 ==================================
__global__ void Mw3T1CoreKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool is_married = false;

  while (!is_married) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    ++w_rank;

    int mj_rank = partner_rank_dview[w_idx];

    while (mj_rank > mi_rank) {
      int observed = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (observed == mj_rank) {
        // success: record where 'mi' should resume if displaced later
        next_proposed_w_dview[mi] = w_rank;

        if (mj_rank == n) {
          // woman was free
          is_married = true;
        } else {
          // displace the previous partner; continue as that man
          mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
          w_rank = next_proposed_w_dview[mi];
        }
        break; // done with CAS loop this iteration
      } else {
        // lost race, re-check woman's current partner rank
        mj_rank = observed;
        if (mj_rank < mi_rank) {
          // she now has someone she prefers more than 'mi'
          break;
        }
      }
    }
  }
}


// ================================= Figrue 7 ==================================
__global__ void Mw3F7CoreKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool is_married = false;

  while (!is_married) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    // advance mi's pointer to the next woman (in case he gets displaced)
    ++w_rank;

    // read current partner rank for w_idx
    int mj_rank = partner_rank_dview[w_idx];

    // If woman prefers current partner strictly more than mi, skip.
    // if (mj_rank < mi_rank) {
    //   continue;
    // }

    // Otherwise, try to install mi as her partner via CAS.
    while (mj_rank > mi_rank) {
      // Attempt to swap mj_rank -> mi_rank
      int observed = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (observed == mj_rank) {
        // Success: record mi's next pointer
        next_proposed_w_dview[mi] = w_rank;

        if (mj_rank == n) {
          // w_idx was free; mi is now married
          is_married = true;
        } else {
          // w_idx dumped her previous partner: let that man continue
          // mi <- previous man index from woman's pref list at rank mj_rank
          mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
          // resume that man's search from where he left off
          w_rank = next_proposed_w_dview[mi];
        }
        break; // break CAS loop; either married or we continue as the displaced man
      } else {
        // Lost the race; re-evaluate with the new observed partner rank
        mj_rank = observed;
        if (mj_rank < mi_rank) {
          // Woman now has someone she prefers more than mi; give up and move on
          break;
        }
      }
    }
  }
}

__global__ void La2F7CoreKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0, mj_rank, mj_rank2;
  PRNode node;
  bool is_married = false;

  while (!is_married) {
    node = prmtx_dview[mi * n + w_rank]; // assuming IDX_MUL_ADD is mi * n + w_rank
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    mj_rank = partner_rank_dview[w_idx];

    if (mj_rank < mi_rank) {
      continue;
    }

    while (mj_rank > mi_rank) {
      mj_rank2 = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (mj_rank2 == mj_rank) {
        next_proposed_w_dview[mi] = w_rank;
        if (mj_rank == n) {
          is_married = true;
        } else if (mj_rank > mi_rank) {
          mi = pref_list_w_dview[w_idx * n + mj_rank];
          w_rank = next_proposed_w_dview[mi];
        }
        break;
      } else {
        mj_rank = mj_rank2;
      }
    }
  }
}

// ================================= Figrue8 Mw3(CAS) ==================================
__global__ void Mw3F8CASTopKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool is_married = false;

  while (!is_married) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    ++w_rank;

    int mj_rank = partner_rank_dview[w_idx];

    while (mj_rank > mi_rank) {
      int observed = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (observed == mj_rank) {
        // success: record where 'mi' should resume if displaced later
        next_proposed_w_dview[mi] = w_rank;

        if (mj_rank == n) {
          // woman was free
          is_married = true;
        } else {
          // displace the previous partner; continue as that man
          mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
          w_rank = next_proposed_w_dview[mi];
        }
        break; // done with CAS loop this iteration
      } else {
        // lost race, re-check woman's current partner rank
        mj_rank = observed;
        if (mj_rank < mi_rank) {
          // she now has someone she prefers more than 'mi'
          break;
        }
      }
    }
  }
}

__global__ void Mw3F8CASBottomKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool is_married = false;

  while (!is_married) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    ++w_rank;

    int mj_rank;

    mj_rank = partner_rank_dview[w_idx];

    mj_rank = atomicCAS(&partner_rank_dview[w_idx], n, mi_rank);

    if (mj_rank < mi_rank) {
      continue;
    } else if (mj_rank == n) {
      break;
    }

    while (mj_rank > mi_rank) {
      int observed = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (observed == mj_rank) {
        // success: record where 'mi' should resume if displaced later
        next_proposed_w_dview[mi] = w_rank;

        if (mj_rank == n) {
          // woman was free
          is_married = true;
        } else {
          // displace the previous partner; continue as that man
          mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
          w_rank = next_proposed_w_dview[mi];
        }
        break; // done with CAS loop this iteration
      } else {
        // lost race, re-check woman's current partner rank
        mj_rank = observed;
        if (mj_rank < mi_rank) {
          // she now has someone she prefers more than 'mi'
          break;
        }
      }
    }
  }
}

// ================================= Figrue8 La2(CAS) ==================================
__global__ void La2F8CASTopKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0, mj_rank, mj_rank2;
  PRNode node;
  bool is_married = false;

  while (!is_married) {
    node = prmtx_dview[mi * n + w_rank]; // assuming IDX_MUL_ADD is mi * n + w_rank
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    mj_rank = partner_rank_dview[w_idx];

    while (mj_rank > mi_rank) {
      mj_rank2 = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (mj_rank2 == mj_rank) {
        next_proposed_w_dview[mi] = w_rank;
        if (mj_rank == n) {
          is_married = true;
        } else if (mj_rank > mi_rank) {
          mi = pref_list_w_dview[w_idx * n + mj_rank];
          w_rank = next_proposed_w_dview[mi];
        }
        break;
      } else {
        mj_rank = mj_rank2;
      }
    }
  }
}

__global__ void La2F8CASBottomKernel(
  int n,
  const PRNode *prmtx_dview,
  int *partner_rank_dview,
  int *next_proposed_w_dview,
  const int *pref_list_w_dview
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0, mj_rank, mj_rank2;
  PRNode node;
  bool is_married = false;

  while (!is_married) {
    node = prmtx_dview[mi * n + w_rank]; // assuming IDX_MUL_ADD is mi * n + w_rank
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    mj_rank = partner_rank_dview[w_idx];

    mj_rank = atomicCAS(&partner_rank_dview[w_idx], n, mi_rank);

    if (mj_rank < mi_rank) {
      continue;
    } else if (mj_rank == n) {
      break;
    }

    while (mj_rank > mi_rank) {
      mj_rank2 = atomicCAS(&partner_rank_dview[w_idx], mj_rank, mi_rank);
      if (mj_rank2 == mj_rank) {
        next_proposed_w_dview[mi] = w_rank;
        if (mj_rank == n) {
          is_married = true;
        } else if (mj_rank > mi_rank) {
          mi = pref_list_w_dview[w_idx * n + mj_rank];
          w_rank = next_proposed_w_dview[mi];
        }
        break;
      } else {
        mj_rank = mj_rank2;
      }
    }
  }
}

// ================================= Figrue8 Mw4(MIN) ==================================
__global__ void Mw4F8BottomKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool paired = false;

  while (!paired) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];
    ++w_rank;

    int mj_rank;
    // If she prefers her current partner (lower rank) over mi, skip.
    // mj_rank = partner_rank_dview[w_idx];
    //
    // if (mj_rank < mi_rank) {
    //   continue;
    // }

    mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);

    if (mj_rank > mi_rank) {
      // We improved her partner (or claimed her if free).
      next_proposed_w_dview[mi] = w_rank;

      if (mj_rank == n) {
        // She was free -> married now.
        paired = true;
      } else {
        // Displaced previous man; continue as that man.
        mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

__global__ void Mw4F8TopLeftKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool paired = false;

  while (!paired) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    ++w_rank;

    // If she prefers her current partner (lower rank) over mi, skip.
    // if (partner_rank_dview[w_idx] < mi_rank) {
    //   continue;
    // }

    int mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);

    if (mj_rank > mi_rank) {
      // We improved her partner (or claimed her if free).
      next_proposed_w_dview[mi] = w_rank;

      if (mj_rank == n) {
        // She was free -> married now.
        paired = true;
      } else {
        // Displaced previous man; continue as that man.
        mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

__global__ void Mw4F8TopRightKernel(
  int n,
  const int *pref_list_m_dview, // [n * n]
  const int *pref_list_w_dview, // [n * n]
  const int *rank_mtx_w_dview, // [n * n]
  int *next_proposed_w_dview, // [n]
  int *partner_rank_dview // [n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_rank = 0;
  bool paired = false;

  while (!paired) {
    // w_idx = pref_list_m[mi * n + w_rank];
    int w_idx = pref_list_m_dview[IDX_MUL_ADD(mi, n, w_rank)];
    // mi_rank = rank_mtx_w[w_idx * n + mi];
    int mi_rank = rank_mtx_w_dview[IDX_MUL_ADD(w_idx, n, mi)];

    ++w_rank;

    // If she prefers her current partner (lower rank) over mi, skip.
    if (partner_rank_dview[w_idx] < mi_rank) {
      continue;
    }

    int mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);

    if (mj_rank > mi_rank) {
      // We improved her partner (or claimed her if free).
      next_proposed_w_dview[mi] = w_rank;

      if (mj_rank == n) {
        // She was free -> married now.
        paired = true;
      } else {
        // Displaced previous man; continue as that man.
        mi = pref_list_w_dview[IDX_MUL_ADD(w_idx, n, mj_rank)];
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

// ================================= Figrue8 La3(MIN) ==================================
__global__ void La3MinF8BottomKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0;
  int mj_rank;
  PRNode node;
  bool paired = false;

  while (!paired) {
    node = prmtx_dview[mi * n + w_rank]; // Replace with IDX_MUL_ADD(mi, n, w_rank) if macro is defined
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    // if (partner_rank_dview[w_idx] < mi_rank) {
    //   continue;
    // }

    mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    if (mj_rank > mi_rank) {
      next_proposed_w_dview[mi] = w_rank;
      if (mj_rank == n) {
        paired = true;
      } else {
        mi = pref_list_w_dview[w_idx * n + mj_rank]; // Replace with IDX_MUL_ADD if needed
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

__global__ void La3MinF8TopLeftKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0;
  int mj_rank;
  PRNode node;
  bool paired = false;

  while (!paired) {
    node = prmtx_dview[mi * n + w_rank]; // Replace with IDX_MUL_ADD(mi, n, w_rank) if macro is defined
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    // if (partner_rank_dview[w_idx] < mi_rank) {
    //   continue;
    // }

    mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    if (mj_rank > mi_rank) {
      next_proposed_w_dview[mi] = w_rank;
      if (mj_rank == n) {
        paired = true;
      } else {
        mi = pref_list_w_dview[w_idx * n + mj_rank]; // Replace with IDX_MUL_ADD if needed
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

__global__ void La3MinF8TopRightKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0;
  int mj_rank;
  PRNode node;
  bool paired = false;

  while (!paired) {
    node = prmtx_dview[mi * n + w_rank]; // Replace with IDX_MUL_ADD(mi, n, w_rank) if macro is defined
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    if (partner_rank_dview[w_idx] < mi_rank) {
      continue;
    }

    mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    if (mj_rank > mi_rank) {
      next_proposed_w_dview[mi] = w_rank;
      if (mj_rank == n) {
        paired = true;
      } else {
        mi = pref_list_w_dview[w_idx * n + mj_rank]; // Replace with IDX_MUL_ADD if needed
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}

// ================================= Figrue 9 ==================================
__global__ void La3F9CoreKernel(
  int n,
  const PRNode *prmtx_dview, // [n * n]
  int *partner_rank_dview, // [n]
  int *next_proposed_w_dview, // [n]
  const int *pref_list_w_dview // [n * n]
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  int mi = tid;
  int w_idx, mi_rank, w_rank = 0;
  int mj_rank;
  PRNode node;
  bool paired = false;

  while (!paired) {
    node = prmtx_dview[mi * n + w_rank]; // Replace with IDX_MUL_ADD(mi, n, w_rank) if macro is defined
    w_idx = node.idx_;
    mi_rank = node.rank_;
    w_rank++;

    if (partner_rank_dview[w_idx] < mi_rank) {
      continue;
    }

    mj_rank = atomicMin(&partner_rank_dview[w_idx], mi_rank);
    if (mj_rank > mi_rank) {
      next_proposed_w_dview[mi] = w_rank;
      if (mj_rank == n) {
        paired = true;
      } else {
        mi = pref_list_w_dview[w_idx * n + mj_rank]; // Replace with IDX_MUL_ADD if needed
        w_rank = next_proposed_w_dview[mi];
      }
    }
  }
}
