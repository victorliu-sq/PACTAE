#include <cuda.h>
#include <memory>
#include <set>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include <smp/smp_engine_multigpu.h>

#include "utils/generate_worklods.h"

#define WORKLOAD_SIZE_MULTIGPU 50000
#define THREADS_PER_BLOCK 64
#define NUM_GPUS 4

void TestMultiGpuCoreProc(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE_MULTIGPU, plM, plW);

  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE_MULTIGPU);

  // Multi GPU engine initialization
  auto engine_multi = MakeUPtr<bamboosmp::SmpEngineMultiGpu>(*smp, WORKLOAD_SIZE_MULTIGPU, NUM_GPUS);
  engine_multi->FindStableMatching();
  auto actual_matching = engine_multi->GetStableMarriage();

  // Check that actual_matching contains all unique indices exactly once
  std::set<int> matched_women_actual;
  bool valid_actual = true;
  for (int woman_idx: actual_matching) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE_MULTIGPU || matched_women_actual.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index in actual_matching: " << woman_idx;
      valid_actual = false;
      break;
    }
    matched_women_actual.insert(woman_idx);
  }
  if (valid_actual && matched_women_actual.size() == WORKLOAD_SIZE_MULTIGPU) {
    LOG(INFO) << "actual_matching correctly contains all unique woman indices.";
  } else if (valid_actual) {
    LOG(ERROR) << "actual_matching does not contain all women indices (size mismatch).";
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
