#include "smp/smp_engine_mw_1.h"

#include <cuda.h>
#include <memory>
#include <smp/gs.cuh>
#include <smp/smp_engine_hybrid.cuh>
#include <glog/logging.h>
#include "utils/generate_worklods.h"
#include <set>

#include "smp_engine_test_utils.h"

int main(int argc, char *argv[]) {
  INIT_GLOG_STR("smp_engine_gs_par_test");


  SmpWorkloadConfig config;
  SetupSmpWorkloadConfig(config);

  TestSmpEngine<bamboosmp::SmpEngineMw>(config);

  SHUTDOWN_GLOG();
}
