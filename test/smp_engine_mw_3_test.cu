#include <glog/logging.h>
#include "smp/smp_engine_mw_3_f8_top.h"
#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_mw_par_gpu_cas_test");


  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineMw3F8Top>(config);

  SHUTDOWN_GLOG();
}
