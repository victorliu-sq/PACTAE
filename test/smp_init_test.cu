#include "smp/smp_init_2.h"

#include "smp_engine_test_utils.h"
#include "smp_init_test_utils.h"
#include "utils/generate_worklods.h"
#include "glog/logging.h"
#include "smp/smp_init_1.h"
#include "smp/smp_init_3.h"

int main() {
  INIT_GLOG_STR("smp_workload_generation");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  // TestSmpInitEngine<bamboosmp::SmpInitEngine1>(config, false);
  //
  // TestSmpInitEngine<bamboosmp::SmpInitEngine1>(config, true);
  //
  // TestSmpInitEngine<bamboosmp::SmpInitEngine2>(config, false);
  //
  // TestSmpInitEngine<bamboosmp::SmpInitEngine2>(config, true);

  TestSmpInitEngine<bamboosmp::SmpInitEngine3>(config, false);

  // TestSmpInitEngine<bamboosmp::SmpInitEngine3>(config, true);

  SHUTDOWN_GLOG();
  return 0;
}
