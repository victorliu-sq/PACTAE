#include "smp_engine_exec.h"
#include "smp_init_engine_exec.h"
#include "smp/smp_engine_gs_1_f7.h"
#include "smp/smp_engine_la_2_f7.h"
#include "smp/smp_engine_la_f7.h"
#include "smp/smp_engine_mw_1_f7.h"
#include "smp/smp_engine_mw_3_f7.h"

int main() {
  INIT_GLOG_STR("figure_7_bottom");
  const size_t workload_size = 30000;
  // const size_t workload_size = 20000;
  // const size_t workload_size = 12000;
  // const size_t workload_size = 10000;
  const size_t group_size = 5;
  // const size_t group_size = 10;
  // ========================================================================================
  // Solo Case
  SmpEngineExecutor<bamboosmp::SmpEngineGsF7>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF7>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLaF7>(SOLO, workload_size);

  // ========================================================================================
  // Congested Case
  SmpEngineExecutor<bamboosmp::SmpEngineGsF7>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF7>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLaF7>(CONGESTED, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F7>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F7>(CONGESTED, workload_size);

  // ========================================================================================
  // Random Case
  SmpEngineExecutor<bamboosmp::SmpEngineGsF7>(RANDOM, workload_size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF7>(RANDOM, workload_size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLaF7>(RANDOM, workload_size, group_size);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F7>(RANDOM, workload_size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F7>(RANDOM, workload_size, group_size);

  SHUTDOWN_GLOG();
}
