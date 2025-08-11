#include <glog/logging.h>
#include "smp/smp_engine_mw_4.h"

#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_mw_par_gpu_test");


  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineMw4>(config);

  SHUTDOWN_GLOG();
}
