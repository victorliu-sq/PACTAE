#ifndef SMP_WORKLOAD_GEN_TEST_UTILS_H
#define SMP_WORKLOAD_GEN_TEST_UTILS_H
#include "utils/generate_worklods.h"

static void TestSmpWorkloadGeneration(WorkloadType workload_type, size_t workload_size, size_t group_size = 5) {
  PreferenceLists plM, plW;
  GenerateWorkloadCached(workload_type, workload_size, plM, plW, group_size);
}

#endif //SMP_WORKLOAD_GEN_TEST_UTILS_H
