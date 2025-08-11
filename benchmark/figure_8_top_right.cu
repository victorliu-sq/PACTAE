#include "smp_engine_exec.h"
#include "smp_init_engine_exec.h"
#include "smp/smp_engine_la_2_f8_top.h"
#include "smp/smp_engine_la_3_f8_tright.h"
#include "smp/smp_engine_mw_3_f8_top.h"
#include "smp/smp_engine_mw_4_f8_tright.h"

int main() {
  INIT_GLOG_STR("figure_8_top_right");

  // ========================================================================================
  // Random Case
  const size_t random_workload_size = 30000;
  // const size_t random_workload_size = 10000;

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 5);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 5);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 5);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 5);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 10);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 10);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 10);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 10);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 20);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 20);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 20);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 20);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 30);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 30);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 30);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 30);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 40);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 40);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 40);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 40);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(RANDOM, random_workload_size, 50);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TRight>(RANDOM, random_workload_size, 50);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(RANDOM, random_workload_size, 50);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TRight>(RANDOM, random_workload_size, 50);

  SHUTDOWN_GLOG();
}
