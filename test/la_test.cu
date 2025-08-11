#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_la_old.cuh>
#include <glog/logging.h>
#include "utils/generate_worklods.h"
#include <set>

#define WORKLOAD_SIZE 10000
#define THREADS_PER_BLOCK 128

void test_single_gpu(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE, plM, plW);

  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE);

  // GS
  auto gs = std::make_unique<bamboosmp::GS>(*smp, THREADS_PER_BLOCK, WORKLOAD_SIZE);
  auto time_gs = gs->StartGS();
  auto match_vec_gs = gs->GetMatchVector();
  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> matched_women_gs;
  bool valid_match = true;
  for (int woman_idx: match_vec_gs) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || matched_women_gs.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    matched_women_gs.insert(woman_idx);
  }

  if (valid_match && matched_women_gs.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "match_vec_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs does not contain all women indices (size mismatch).";
  }

  // La
  auto start_time_bamboo = getNanoSecond();
  auto la = std::make_unique<bamboosmp::SmpEngineLaOld>(*smp, WORKLOAD_SIZE);
  la->FindStableMatching();
  auto end_time_bamboo = getNanoSecond();
  auto match_vec_bsmp = la->GetStableMarriage();
  auto time_bamboo = (end_time_bamboo - start_time_bamboo) / 1e6;

  // Check match_vec_la (BambooSMP) for correctness
  std::set<int> matched_women_la;
  bool valid_match_la = true;
  for (int woman_idx: match_vec_bsmp) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || matched_women_la.count(woman_idx)) {
      LOG(ERROR) << "[BambooSMP] Invalid or duplicate woman index found: " << woman_idx;
      valid_match_la = false;
      break;
    }
    matched_women_la.insert(woman_idx);
  }

  if (valid_match_la && matched_women_la.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "[BambooSMP] match_vec_la correctly contains all unique woman indices.";
  } else if (valid_match_la) {
    LOG(ERROR) << "[BambooSMP] match_vec_la does not contain all women indices (size mismatch).";
  }

  bool correct = (match_vec_bsmp == match_vec_gs);

  if (correct) {
    // LOG(INFO) << "Correct!";
    std::cout << "Correct!" << std::endl;
  } else {
    LOG(ERROR) << "Wrong :( Results do not match.";
    for (size_t i = 0; i < WORKLOAD_SIZE; ++i) {
      if (match_vec_bsmp[i] != match_vec_gs[i]) {
        LOG(ERROR) << "Mismatch at man " << i << ": BambooSMP paired with woman " << match_vec_bsmp[i]
            << ", GS paired with woman " << match_vec_gs[i];
      }
    }
  }

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
  // std::thread terminator(ForcedTerminationThread, 20);
  // terminator.detach();
  // Easily switch workloads here
  test_single_gpu(CONGESTED);
  // test_single_gpu(RANDOM);
  // test_single_gpu(SOLO);

  SHUTDOWN_GLOG();
}
