#include <glog/logging.h>
#include "smp_engine_test_utils.h"
#include "smp/smp_engine_gs_1_f7.h"
#include "smp/smp_engine_la_2_f7.h"
#include "smp/smp_engine_la_f7.h"
#include "smp/smp_engine_mw_1_f7.h"
#include "smp/smp_engine_mw_3_f7.h"

int main() {
  INIT_GLOG_STR("smp_engine_f7_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  // TestSmpEngine<bamboosmp::SmpEngineGsF7>(config);
  // TestSmpEngine<bamboosmp::SmpEngineMwF7>(config);

  // TestSmpEngine<bamboosmp::SmpEngineLaF7>(config);

  TestSmpEngine<bamboosmp::SmpEngineMw3F7>(config);

  TestSmpEngine<bamboosmp::SmpEngineLa2F7>(config);

  SHUTDOWN_GLOG();
}
