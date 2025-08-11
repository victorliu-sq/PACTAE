#ifndef SMP_INIT_ENGINE_EXEC_H
#define SMP_INIT_ENGINE_EXEC_H

#include "utils/generate_workloads_flat.h"
#include "smp/smp_init.h"

template<typename SmpInitEngine>
static void SmpInitEngineExecutor(WorkloadType workload_type, size_t workload_size, bool init_pr_mtx, size_t group_size = 5) {
  // PreferenceLists plM, plW;
  // GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);
  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  FlatPreferenceList plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  // auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);
  auto smp = bamboosmp::SmpObjInit::CreateFromFlatLists(plM, plW, workload_size);

  // Execute SmpEngine
  SmpInitEngine smp_init_engine(*smp, workload_size,init_pr_mtx);

  // ============================= Log this execution ======================================
  OutStringStream oss;
  oss << "Executing SmpInitEngine" << smp_init_engine.GetEngineName()
            << " with workload: " << WorkloadTypeToString(workload_type)
            << ", size: " << workload_size
            << ((workload_type == RANDOM) ? (", group_size: " + std::to_string(group_size)) : "");
  String log_info = oss.str();

  std::cout << log_info << std::endl;
  LOG(INFO) << log_info;
  // ========================================================================================

  // Run the SmpInitEngine on the smp workload
  smp_init_engine.Init();
  smp_init_engine.PrintProfilingInfo();
}


#endif //SMP_INIT_ENGINE_EXEC_H
