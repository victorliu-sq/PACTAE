#include <glog/logging.h>
#include "smp/smp_engine_mw_2.h"

#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_mw_par_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineMw2>(config);

  SHUTDOWN_GLOG();
}
