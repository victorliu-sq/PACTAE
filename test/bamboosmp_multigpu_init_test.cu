#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include <smp/smp_engine_multigpu.h>

#include "utils/generate_worklods.h"

#define WORKLOAD_SIZE_MULTIGPU 1000
#define THREADS_PER_BLOCK 64
#define NUM_GPUS 1

// Compare PRMatrices clearly and report the first mismatch
bool ComparePRMatrices(const std::vector<PRNode> &expected_prmatrix,
                       const std::vector<PRNode> &actual_prmatrix, int n) {
  for (size_t i = 0; i < expected_prmatrix.size(); ++i) {
    if (expected_prmatrix[i].idx_ != actual_prmatrix[i].idx_ ||
        expected_prmatrix[i].rank_ != actual_prmatrix[i].rank_) {
      LOG(ERROR) << "Mismatch at [" << i / n << "][" << i % n << "]: "
          << "Expected(Single)(idx: " << expected_prmatrix[i].idx_ << ", rank: " << expected_prmatrix[i].rank_ << "), "
          << "Actual(Multi)(idx: " << actual_prmatrix[i].idx_ << ", rank: " << actual_prmatrix[i].rank_ << ")";
      return false;
    }
  }
  return true;
}

bool CompareRankMatrices(const std::vector<int> &expected_rank_mtx,
                         const std::vector<int> &actual_rank_mtx, int n) {
  for (size_t i = 0; i < expected_rank_mtx.size(); ++i) {
    if (expected_rank_mtx[i] != actual_rank_mtx[i]) {
      LOG(ERROR) << "RankMatrix mismatch at [" << i / n << "][" << i % n << "]: "
          << "Expected(Single): " << expected_rank_mtx[i] << ", "
          << "Actual(Multi): " << actual_rank_mtx[i];
      return false;
    }
  }
  return true;
}

void TestMultiGpuCoreProc(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE_MULTIGPU, plM, plW);

  auto smp_single = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE_MULTIGPU);
  auto smp_multi = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE_MULTIGPU);

  // Single GPU engine initialization
  auto engine_single = MakeUPtr<bamboosmp::SmpEngineHybrid>(*smp_single, WORKLOAD_SIZE_MULTIGPU);
  engine_single->RunInitProcOnlyForTest();

  // Multi GPU engine initialization
  auto engine_multi = MakeUPtr<bamboosmp::SmpEngineMultiGpu>(*smp_multi, WORKLOAD_SIZE_MULTIGPU, NUM_GPUS);
  engine_multi->RunInitProcOnlyForTest();

  // Compare RankMatrices for correctness
  auto expected_rank_mtx = engine_single->RetrieveRankMatrixForTest();
  auto actual_rank_mtx = engine_multi->RetrieveRankMatrixForTest();
  if (CompareRankMatrices(expected_rank_mtx, actual_rank_mtx, WORKLOAD_SIZE_MULTIGPU)) {
    LOG(INFO) << "RankMatrices match exactly.";
  } else {
    LOG(ERROR) << "RankMatrix mismatch detected explicitly!";
  }

  // Compare PRMatrices for correctness
  auto actual_prmatrix = engine_multi->RetrievePRMatrixForTest();
  auto expected_prmatrix = engine_single->RetrievePRMatrixForTest();
  if (ComparePRMatrices(expected_prmatrix, actual_prmatrix, WORKLOAD_SIZE_MULTIGPU)) {
    LOG(INFO) << "Success: PRMatrix from Single-GPU and Multi-GPU match!";
  } else {
    LOG(ERROR) << "Error: PRMatrix mismatch detected!";
  }
}

int main(int argc, char *argv[]) {
  INIT_GLOG_ARGS();
  LOG(INFO) << "Testing the BambooSMP-MultiGpu Version.";

  // Easily switch workloads here
  TestMultiGpuCoreProc(CONGESTED);
  // TestMultiGpuInitProc(RANDOM);
  // TestMultiGpuInitProc(SOLO);

  SHUTDOWN_GLOG();
}
