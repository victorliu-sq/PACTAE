#include <glog/logging.h>
#include "smp_engine_test_utils.h"
#include "smp/smp_engine_mw_3_f8_bottom.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_mw_f8_test");


  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineMw3F8Bottom>(config);

  SHUTDOWN_GLOG();
}
