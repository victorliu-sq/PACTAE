#include "smp/smp_engine_la_f7.h"

#include <glog/logging.h>
#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_la_seq_cpu_test");

  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineLaF7>(config);

  SHUTDOWN_GLOG();
}
