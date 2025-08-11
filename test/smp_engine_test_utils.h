#ifndef SMP_ENGINE_TEST_UTILS_H
#define SMP_ENGINE_TEST_UTILS_H

#include <set>
#include "smp/smp.h"
#include "smp/smp_engine_bamboo.h"
#include "smp/smp_engine_gs_1.h"
#include "smp/smp_engine_mw_1.h"
#include "utils/generate_workloads_flat.h"

struct SmpWorkloadConfig {
  WorkloadType workload_type;
  size_t workload_size;
  size_t group_size;
};

static void SetupSmpWorkloadConfig(SmpWorkloadConfig &config) {
  // config.workload_type = SOLO;
  config.workload_type = CONGESTED;
  // config.workload_type = RANDOM;

  // config.workload_size = 30000;
  // config.workload_size = 20000;
  // config.workload_size = 15000;
  // config.workload_size = 10000;
  // config.workload_size = 5000;
  config.workload_size = 1000;
  // config.workload_size = 10;
  // config.workload_size = 2;

  config.group_size = 5;
}

template<typename SmpEngine>
static void TestSmpEngine(SmpWorkloadConfig config) {
  WorkloadType workload_type = config.workload_type;
  size_t workload_size = config.workload_size;
  size_t group_size = config.group_size;

  // PreferenceLists plM, plW;
  FlatPreferenceList plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  auto smp = bamboosmp::SmpObj::CreateFromFlatLists(plM, plW, workload_size);

  // GS
  auto gs_seq = bamboosmp::SmpEngineGs(*smp, workload_size);
  auto match_vec_gs_seq = gs_seq.FindStableMatching();
  gs_seq.PrintProfilingInfo();
  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> expected_match;
  bool valid_match = true;
  for (int woman_idx: match_vec_gs_seq) {
    if (woman_idx < 0 || woman_idx >= workload_size || expected_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    expected_match.insert(woman_idx);
  }

  if (valid_match && expected_match.size() == workload_size) {
    LOG(INFO) << "match_vec_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs does not contain all women indices (size mismatch).";
  }

  // SmpEngineGsSeq
  SmpEngine smp_engine_gs_seq(*smp, workload_size);
  auto match_vec_smp_engine_gs_seq = smp_engine_gs_seq.FindStableMatching();
  smp_engine_gs_seq.PrintProfilingInfo();

  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> actual_match;
  valid_match = true;
  for (int woman_idx: match_vec_smp_engine_gs_seq) {
    if (woman_idx < 0 || woman_idx >= workload_size || actual_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    actual_match.insert(woman_idx);
  }

  if (valid_match && actual_match.size() == workload_size) {
    LOG(INFO) << "match_vec_gs_par_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs_par_gs does not contain all women indices (size mismatch).";
  }

  bool correct = (match_vec_gs_seq == match_vec_smp_engine_gs_seq);

  if (correct) {
    // LOG(INFO) << "Correct!";
    std::cout << "Correct!" << std::endl;
  } else {
    LOG(ERROR) << "Wrong :( Results do not match.";
    for (size_t i = 0; i < workload_size; ++i) {
      if (match_vec_smp_engine_gs_seq[i] != match_vec_gs_seq[i]) {
        LOG(ERROR) << "Mismatch at man " << i << ": GS Par CPU paired with woman " << match_vec_smp_engine_gs_seq[i]
            << ", GS paired with woman " << match_vec_gs_seq[i];
      }
    }
  }
}

template<typename SmpEngineContrast=bamboosmp::SmpEngineGs>
static void TestSmpEngineBamboo(SmpWorkloadConfig config) {
  WorkloadType workload_type = config.workload_type;
  size_t workload_size = config.workload_size;
  size_t group_size = config.group_size;

  // PreferenceLists plM, plW;
  FlatPreferenceList plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  auto smp = bamboosmp::SmpObj::CreateFromFlatLists(plM, plW, workload_size);

  // GS
  auto gs_seq = SmpEngineContrast(*smp, workload_size);
  auto match_vec_gs_seq = gs_seq.FindStableMatching();
  gs_seq.PrintProfilingInfo();
  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> expected_match;
  bool valid_match = true;
  for (int woman_idx: match_vec_gs_seq) {
    if (woman_idx < 0 || woman_idx >= workload_size || expected_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    expected_match.insert(woman_idx);
  }

  if (valid_match && expected_match.size() == workload_size) {
    LOG(INFO) << "match_vec_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs does not contain all women indices (size mismatch).";
  }

  // SmpEngineGsSeq
  bamboosmp::SmpEngineBamboo smp_engine_gs_seq(*smp, workload_size);
  auto match_vec_smp_engine_gs_seq = smp_engine_gs_seq.FindStableMatching();
  smp_engine_gs_seq.PrintProfilingInfo();

  // Test unique woman indices

  // Check that match_vec_gs contains all unique indices from 0 to n-1 exactly once
  std::set<int> actual_match;
  valid_match = true;
  for (int woman_idx: match_vec_smp_engine_gs_seq) {
    if (woman_idx < 0 || woman_idx >= workload_size || actual_match.count(woman_idx)) {
      LOG(ERROR) << "Invalid or duplicate woman index found: " << woman_idx;
      valid_match = false;
      break;
    }
    actual_match.insert(woman_idx);
  }

  if (valid_match && actual_match.size() == workload_size) {
    LOG(INFO) << "match_vec_gs_par_gs correctly contains all unique woman indices.";
  } else if (valid_match) {
    LOG(ERROR) << "match_vec_gs_par_gs does not contain all women indices (size mismatch).";
  }

  bool correct = (match_vec_gs_seq == match_vec_smp_engine_gs_seq);

  if (correct) {
    // LOG(INFO) << "Correct!";
    std::cout << "Correct!" << std::endl;
  } else {
    LOG(ERROR) << "Wrong :( Results do not match.";
    for (size_t i = 0; i < workload_size; ++i) {
      if (match_vec_smp_engine_gs_seq[i] != match_vec_gs_seq[i]) {
        LOG(ERROR) << "Mismatch at man " << i << ": GS Par CPU paired with woman " << match_vec_smp_engine_gs_seq[i]
            << ", GS paired with woman " << match_vec_gs_seq[i];
      }
    }
  }

  SHUTDOWN_GLOG();
  std::_Exit(0);
}

#endif //SMP_ENGINE_TEST_UTILS_H
