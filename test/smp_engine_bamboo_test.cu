#include "smp/smp_engine_bamboo.h"

#include <glog/logging.h>
#include "smp_engine_test_utils.h"
#include "smp/smp_engine_gs_2.h"
#include "smp/smp_engine_gs_2_f9.h"
#include "smp/smp_engine_mw_3_f8_top.h"

int main() {
  INIT_GLOG_STR("smp_engine_bamboo_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngineBamboo<bamboosmp::SmpEngineMw3F8Top>(config);
}
