#include "smp_engine_exec.h"
#include "smp/smp_engine_gs_2.h"
#include "smp/smp_engine_mw_1.h"
#include "smp/smp_engine_mw_2.h"
#include "smp/smp_engine_mw_3_f8_top.h"
#include "smp/smp_engine_mw_3_t1.h"

int main() {
  INIT_GLOG_STR("table_1");

  // const size_t workload_size = 10000;
  const size_t workload_size = 8000;
  SmpEngineExecutor<bamboosmp::SmpEngineGs>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineGs2>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw2>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw3T1>(SOLO, workload_size);

  SHUTDOWN_GLOG();
}
