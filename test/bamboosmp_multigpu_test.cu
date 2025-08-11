#include <cuda.h>
#include <memory>
#include <set>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include <smp/smp_engine_multigpu.h>

#include "utils/generate_worklods.h"

#define WORKLOAD_SIZE 10000
#define THREADS_PER_BLOCK 128
#define NUM_GPUS 1

void test_multigpu(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE, plM, plW);

  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE);

  auto gs = std::make_unique<bamboosmp::GS>(*smp, THREADS_PER_BLOCK, WORKLOAD_SIZE);
  auto time_gs = gs->StartGS();
  auto expected_matching = gs->GetMatchVector();

  auto bamboo = std::make_unique<bamboosmp::SmpEngineMultiGpu>(*smp, WORKLOAD_SIZE, NUM_GPUS);

  auto start_time_bamboo = getNanoSecond();
  bamboo->FindStableMatching();
  auto end_time_bamboo = getNanoSecond();
  auto actual_matching = bamboo->GetStableMarriage();
  auto time_bamboo = (end_time_bamboo - start_time_bamboo) / 1e6;

  // Check that expected_matching contains all unique indices exactly once
  std::set<int> matched_women_expected;
  bool valid_expected = true;
  for (int woman_idx: expected_matching) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || matched_women_expected.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index in expected_matching: " << woman_idx;
      valid_expected = false;
      break;
    }
    matched_women_expected.insert(woman_idx);
  }
  if (valid_expected && matched_women_expected.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "expected_matching correctly contains all unique woman indices.";
  } else if (valid_expected) {
    LOG(ERROR) << "expected_matching does not contain all women indices (size mismatch).";
  }

  // Check that actual_matching contains all unique indices exactly once
  std::set<int> matched_women_actual;
  bool valid_actual = true;
  for (int woman_idx: actual_matching) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || matched_women_actual.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index in actual_matching: " << woman_idx;
      valid_actual = false;
      break;
    }
    matched_women_actual.insert(woman_idx);
  }
  if (valid_actual && matched_women_actual.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "actual_matching correctly contains all unique woman indices.";
  } else if (valid_actual) {
    LOG(ERROR) << "actual_matching does not contain all women indices (size mismatch).";
  }


  // Existing correctness check between expected and actual
  for (int i = 0; i < WORKLOAD_SIZE; ++i) {
    CHECK_EQ(expected_matching[i], actual_matching[i])
        << "Mismatch at index " << i << ": SingleGPU(Expected)=" << expected_matching[i]
        << ", MultiGPU(Actual)=" << actual_matching[i];
  }
  LOG(INFO) << "TestMultiCoreProc: Matching correctness verified.";

  LOG(INFO) << "GS time: " << time_gs << " ms";
  LOG(INFO) << "Bamboo time: " << time_bamboo << " ms";
}

CUcontext globalCtx = nullptr;

void InitializeCudaContext() {
  cudaFree(0); // ensures CUDA context creation
  cuCtxGetCurrent(&globalCtx);
  if (!globalCtx) {
    LOG(ERROR) << "Main thread failed to get CUDA context.";
    std::exit(1);
  }
  LOG(INFO) << "CUDA context initialized and stored.";
}

void ForceTerminateCudaContext() {
  if (globalCtx) {
    cuCtxSetCurrent(globalCtx); // Set context explicitly for this thread
    LOG(WARNING) << "Force killing current CUDA kernel and destroying CUDA context.";
    cuCtxDestroy(globalCtx); // Force immediate termination
    LOG(INFO) << "CUDA context forcibly terminated.";
  } else {
    LOG(ERROR) << "CUDA context was not initialized or lost.";
  }
}

void ForcedTerminationThread(int timeout_seconds) {
  std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));
  LOG(WARNING) << "Forcing termination after timeout.";
  ForceTerminateCudaContext();
  std::_Exit(0);
}


int main(int argc, char *argv[]) {
  INIT_GLOG_ARGS();
  LOG(INFO) << "Testing the BambooSMP.";

  InitializeCudaContext(); // Call immediately at start

  // Launch forced termination thread (set timeout to ~10 seconds as example)
  std::thread terminator(ForcedTerminationThread, 500);
  terminator.detach();
  // Easily switch workloads here
  // test_multigpu(CONGESTED);
  // test_multigpu(RANDOM);
  test_multigpu(SOLO);

  SHUTDOWN_GLOG();
}
