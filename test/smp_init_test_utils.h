#ifndef SMP_INIT_TEST_UTILS_H
#define SMP_INIT_TEST_UTILS_H
#include "smp_engine_test_utils.h"

template<typename SmpInitEngine>
static void TestSmpInitEngine(SmpWorkloadConfig config, bool init_pr_mtx) {
  WorkloadType workload_type = config.workload_type;
  size_t workload_size = config.workload_size;
  size_t group_size = config.group_size;

  PreferenceLists plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);

  auto smp = bamboosmp::SmpObj::CreateFromPrefLists(plM, plW, workload_size);

  SmpInitEngine smp_init_engine(*smp, workload_size, init_pr_mtx);
  smp_init_engine.Init();
  smp_init_engine.PrintProfilingInfo();
}

#endif //SMP_INIT_TEST_UTILS_H
