#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include <smp/smp_engine_multigpu.h>

#include "utils/generate_worklods.h"

#define WORKLOAD_SIZE_MULTIGPU 1000
#define THREADS_PER_BLOCK 64
#define NUM_GPUS 4


inline void TestMultiGpuCoreProc(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE_MULTIGPU, plM, plW);

  auto smp_single = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE_MULTIGPU);
  auto smp_multi = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE_MULTIGPU);

  // Single GPU engine initialization
  auto engine_single = MakeUPtr<bamboosmp::SmpEngineHybrid>(*smp_single, WORKLOAD_SIZE_MULTIGPU);
  engine_single->FindStableMatching();

  // Multi GPU engine initialization
  auto engine_multi = MakeUPtr<bamboosmp::SmpEngineMultiGpu>(*smp_multi, WORKLOAD_SIZE_MULTIGPU, NUM_GPUS);
  engine_multi->FindStableMatching();

  auto expected_matching = engine_single->GetStableMarriage();
  auto actual_matching = engine_multi->GetStableMarriage();

  for (int i = 0; i < WORKLOAD_SIZE_MULTIGPU; ++i) {
    CHECK_EQ(expected_matching[i], actual_matching[i])
        << "Mismatch at index " << i << ": SingleGPU(Expected)=" << expected_matching[i]
        << ", MultiGPU(Actual)=" << actual_matching[i];
  }

  LOG(INFO) << "TestMultiGpuCoreProc: Matching correctness verified.";
}

int main(int argc, char *argv[]) {
  INIT_GLOG_ARGS();
  LOG(INFO) << "Testing the BambooSMP-MultiGpu Version";

  // Easily switch workloads here
  TestMultiGpuCoreProc(CONGESTED);
  // TestMultiGpuCoreProc(RANDOM);
  // TestMultiGpuCoreProc(SOLO);

  SHUTDOWN_GLOG();
}
