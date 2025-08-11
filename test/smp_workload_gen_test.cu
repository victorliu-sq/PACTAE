#include "smp_workload_gen_test_utils.h"
#include "smp/smp_engine_bamboo.h"
#include "utils/generate_worklods.h"

int main() {
  INIT_GLOG_STR("smp_generate_worklods");

  RemoveFile("data/workloads/smp_workload_congested_10.txt");
  RemoveFile("data/workloads/smp_workload_random_10_3.txt");
  RemoveFile("data/workloads/smp_workload_solo_10.txt");
  RemoveFile("data/workloads/smp_workload_perfect_10.txt");
  assert(!FileExists("data/workloads/smp_workload_congested_10.txt"));
  assert(!FileExists("data/workloads/smp_workload_random_10_3.txt"));
  assert(!FileExists("data/workloads/smp_workload_solo_10.txt"));

  TestSmpWorkloadGeneration(PERFECT, 10);
  TestSmpWorkloadGeneration(SOLO, 10);
  TestSmpWorkloadGeneration(CONGESTED, 10);
  TestSmpWorkloadGeneration(RANDOM, 10, 3);

  assert(FileExists("data/workloads/smp_workload_perfect_10.txt"));
  assert(FileExists("data/workloads/smp_workload_congested_10.txt"));
  assert(FileExists("data/workloads/smp_workload_random_10_3.txt"));
  assert(FileExists("data/workloads/smp_workload_solo_10.txt"));
  SHUTDOWN_GLOG();
}
