#include "smp/smp_engine_la_3.h"

#include <glog/logging.h>
#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_la_par_gpu_cas_test");


  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineLa3>(config);

  SHUTDOWN_GLOG();
}
