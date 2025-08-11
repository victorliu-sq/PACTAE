#include "utils/generate_worklods.h"
#include "glog/logging.h"
#include <iostream>

void GenerateSpecificWorkload(WorkloadType type, size_t n, int group_size = 5) {
  PreferenceLists plM, plW;
  GenerateWorkloadCached(type, n, plM, plW, group_size);

  String workload_type_str;
  switch (type) {
    case SOLO: workload_type_str = "SOLO";
      break;
    case CONGESTED: workload_type_str = "CONGESTED";
      break;
    case RANDOM: workload_type_str = "RANDOM";
      break;
    default: workload_type_str = "UNKNOWN";
      break;
  }

  String message = "Generated workload: " + workload_type_str +
                   ", size: " + std::to_string(n) +
                   (type == RANDOM ? (", group size: " + std::to_string(group_size)) : "");

  std::cout << message << std::endl;
  LOG(INFO) << message;
}

int main() {
  INIT_GLOG_STR("smp_workload_generation");

  // Solo Workloads
  // GenerateSpecificWorkload(SOLO, 10000);
  // GenerateSpecificWorkload(SOLO, 30000);
  for (size_t size: {10000, 30000}) {
    GenerateSpecificWorkload(SOLO, size);
  }

  // Congested Workloads
  for (size_t size: {5000, 10000, 15000, 20000, 25000, 30000}) {
    GenerateSpecificWorkload(CONGESTED, size);
  }

  // Random Workloads
  for (int group_size: {1, 5, 10, 20, 30, 40, 50}) {
    GenerateSpecificWorkload(RANDOM, 30000, group_size);
  }

  SHUTDOWN_GLOG();
  return 0;
}
