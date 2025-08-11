#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include "utils/generate_worklods.h"
#include <set>
#include <signal.h>
#include <unistd.h>

#define WORKLOAD_SIZE 12000
#define THREADS_PER_BLOCK 128

static inline void test_single_gpu(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE, plM, plW);
  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE);

  // GS
  auto gs = std::make_unique<bamboosmp::GS>(*smp, THREADS_PER_BLOCK, WORKLOAD_SIZE);

  auto time_gs = gs->StartGS();
  auto match_vec_gs = gs->GetMatchVector();

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

  cudaFree(0); // Explicitly create CUDA context early to minimize the allocation time
  // BSMP
  auto bamboo = std::make_unique<bamboosmp::SmpEngineHybrid>(*smp, WORKLOAD_SIZE);

  auto start_time_bamboo = getNanoSecond();
  bamboo->FindStableMatching();
  auto end_time_bamboo = getNanoSecond();
  auto match_vec_bsmp = bamboo->GetStableMarriage();
  auto time_bamboo = (end_time_bamboo - start_time_bamboo) / 1e6;

  bool correct = match_vec_bsmp == match_vec_gs;

  // Check match_vec_la (BambooSMP) for correctness
  std::set<int> matched_women_bsmp;
  bool valid_match_bsmp = true;
  for (int woman_idx: match_vec_bsmp) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || matched_women_bsmp.count(woman_idx)) {
      LOG(ERROR) << "[BambooSMP] Invalid or duplicate woman index found: " << woman_idx;
      valid_match_bsmp = false;
      break;
    }
    matched_women_bsmp.insert(woman_idx);
  }

  if (valid_match_bsmp && matched_women_bsmp.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "[BambooSMP] match_vec_la correctly contains all unique woman indices.";
  } else if (valid_match_bsmp) {
    LOG(ERROR) << "[BambooSMP] match_vec_la does not contain all women indices (size mismatch).";
  }

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
  LOG(INFO) << "GS time: " << +time_gs << " ms";
  std::cout << "GS time: " << +time_gs << " ms" << std::endl;
  LOG(INFO) << "Bamboo time: " << time_bamboo << " ms";
  std::cout << "Bamboo time: " << time_bamboo << " ms" << std::endl;

  pid_t pid = fork();
  if (pid > 0) {
    kill(getpid(), SIGKILL);
  } else {
    std::cout << "This is children process after forking. " << std::endl;
    SHUTDOWN_GLOG();
    _exit(0); // Do NOT run destructors or cleanup CUDA
  }
}

int main(int argc, char *argv[]) {
  cudaFree(0); // Explicitly create CUDA context early to minimize the allocation time
  INIT_GLOG_ARGS();
  LOG(INFO) << "Testing the BambooSMP.";

  // Easily switch workloads here
  // test_single_gpu(CONGESTED);
  // test_single_gpu(RANDOM);
  test_single_gpu(SOLO);
}
