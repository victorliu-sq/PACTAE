#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include "utils/generate_worklods.h"
#include <set>

#define WORKLOAD_SIZE 10000
#define THREADS_PER_BLOCK 128

void test_gs_seq_cpu(WorkloadType workload_type) {
  PreferenceLists plM, plW;
  GenerateWorkload(workload_type, WORKLOAD_SIZE, plM, plW);

  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, WORKLOAD_SIZE);

  // GS
  auto gs_seq = std::make_unique<bamboosmp::GS>(*smp, THREADS_PER_BLOCK, WORKLOAD_SIZE);
  auto time_gs = gs_seq->StartGS();
  auto match_vec_gs_seq = gs_seq->GetMatchVector();
  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> expected_match;
  bool valid_match = true;
  for (int woman_idx: match_vec_gs_seq) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || expected_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    expected_match.insert(woman_idx);
  }

  if (valid_match && expected_match.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "match_vec_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs does not contain all women indices (size mismatch).";
  }

  // ParGs
  auto gs_par_cpu = std::make_unique<bamboosmp::GS>(*smp, THREADS_PER_BLOCK, WORKLOAD_SIZE);
  auto time_gs_par_cpu = gs_par_cpu->StartGSParallel();
  auto match_vec_gs_par_cpu = gs_par_cpu->GetMatchVectorParallelCPU();
  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> actual_match;
  valid_match = true;
  for (int woman_idx: match_vec_gs_par_cpu) {
    if (woman_idx < 0 || woman_idx >= WORKLOAD_SIZE || actual_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    actual_match.insert(woman_idx);
  }

  if (valid_match && actual_match.size() == WORKLOAD_SIZE) {
    LOG(INFO) << "match_vec_gs_par_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs_par_gs does not contain all women indices (size mismatch).";
  }

  bool correct = (match_vec_gs_seq == match_vec_gs_par_cpu);

  if (correct) {
    // LOG(INFO) << "Correct!";
    std::cout << "Correct!" << std::endl;
  } else {
    LOG(ERROR) << "Wrong :( Results do not match.";
    for (size_t i = 0; i < WORKLOAD_SIZE; ++i) {
      if (match_vec_gs_par_cpu[i] != match_vec_gs_seq[i]) {
        LOG(ERROR) << "Mismatch at man " << i << ": GS Par CPU paired with woman " << match_vec_gs_par_cpu[i]
            << ", GS paired with woman " << match_vec_gs_seq[i];
      }
    }
  }

  LOG(INFO) << "GS-Seq time: " << time_gs << " ms";
  LOG(INFO) << "GS-Par-Cpu time: " << time_gs_par_cpu << " ms";
}

int main(int argc, char *argv[]) {
  INIT_GLOG_ARGS();
  LOG(INFO) << "Testing the GS PAR CPU.";

  // test_gs_par_cpu(CONGESTED);
  test_gs_seq_cpu(RANDOM);
  // test_gs_par_cpu(SOLO);

  SHUTDOWN_GLOG();
}
