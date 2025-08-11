#include "smp/smp_engine_gs_1_f9.h"

#include <glog/logging.h>
#include "smp/smp_engine_mw_2.h"

#include "smp_engine_test_utils.h"

int main() {
  INIT_GLOG_STR("smp_engine_gs_f9_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineGsF9>(config);

  SHUTDOWN_GLOG();
}
