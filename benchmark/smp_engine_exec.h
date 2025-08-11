#ifndef SMP_ENGINE_RUNNER_H
#define SMP_ENGINE_RUNNER_H
#include "smp/smp.h"
#include "utils/generate_workloads_flat.h"

template<typename SmpEngine>
static void SmpEngineExecutor(WorkloadType workload_type, size_t workload_size, size_t group_size = 5) {
  // PreferenceLists plM, plW;
  FlatPreferenceList plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  auto smp = bamboosmp::SmpObj::CreateFromFlatLists(plM, plW, workload_size);

  // Execute SmpEngine
  SmpEngine smp_engine(*smp, workload_size);

  // Log this execution
  OutStringStream oss;
  oss << "Executing " << smp_engine.GetEngineName()
      << " with workload: " << WorkloadTypeToString(workload_type)
      << ", size: " << workload_size
      << ((workload_type == RANDOM) ? (", group_size: " + std::to_string(group_size)) : "");
  String log_info = oss.str();

  std::cout << log_info << std::endl;
  LOG(INFO) << log_info;

  // run the SmpEngine on the smp workload
  auto match_vec_gs_seq = smp_engine.FindStableMatching();
  smp_engine.PrintProfilingInfo();
}

template<typename SmpEngine>
static void SmpEngineExecutorTermination(WorkloadType workload_type, size_t workload_size, size_t group_size = 5) {
  // PreferenceLists plM, plW;
  // GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);
  //
  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  FlatPreferenceList plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  auto smp = bamboosmp::SmpObj::CreateFromFlatLists(plM, plW, workload_size);

  // Execute SmpEngine
  SmpEngine smp_engine(*smp, workload_size);

  // Log this execution
  OutStringStream oss;
  oss << "Executing " << smp_engine.GetEngineName()
      << " with workload: " << WorkloadTypeToString(workload_type)
      << ", size: " << workload_size
      << ((workload_type == RANDOM) ? (", group_size: " + std::to_string(group_size)) : "");
  String log_info = oss.str();

  std::cout << log_info << std::endl;
  LOG(INFO) << log_info;

  // run the SmpEngine on the smp workload
  auto match_vec_gs_seq = smp_engine.FindStableMatching();
  smp_engine.PrintProfilingInfo();

  SHUTDOWN_GLOG();
  std::_Exit(0);
}


#endif //SMP_ENGINE_RUNNER_H
