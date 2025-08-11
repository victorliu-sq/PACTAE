#include <glog/logging.h>

#include "smp_engine_test_utils.h"
#include "smp/smp_engine_mw_1_f9.h"
#include "smp/smp_engine_mw_2_f9.h"
#include "smp/smp_engine_mw_3_f9.h"

int main() {
  INIT_GLOG_STR("smp_engine_mw_f9_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  // TestSmpEngine<bamboosmp::SmpEngineMwF9>(config);

  // TestSmpEngine<bamboosmp::SmpEngineMw2F9>(config);

  TestSmpEngine<bamboosmp::SmpEngineMw3F9>(config);

  SHUTDOWN_GLOG();
}
