#include "smp_engine_exec.h"
#include "smp/smp_engine_gs_1_f3.h"
#include "smp/smp_engine_mw_1_f3.h"

int main() {
  INIT_GLOG_STR("figure_3");

  // const size_t workload_size = 15000;
  // const size_t workload_size = 7000;
  const size_t workload_size = 5000;
  SmpEngineExecutor<bamboosmp::SmpEngineGsF3>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw1F3>(SOLO, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineGsF3>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw1F3>(CONGESTED, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineGsF3>(RANDOM, workload_size, 5);
  SmpEngineExecutor<bamboosmp::SmpEngineMw1F3>(RANDOM, workload_size, 5);

  SHUTDOWN_GLOG();
}
