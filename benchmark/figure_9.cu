#include <cuda.h>

#include "smp_engine_exec.h"
#include "smp_init_engine_exec.h"
#include "smp/smp_engine_bamboo.h"
#include "smp/smp_engine_gs_1_f9.h"
#include "smp/smp_engine_gs_2.h"
#include "smp/smp_engine_gs_2_f9.h"
#include "smp/smp_engine_mw_1.h"
#include "smp/smp_engine_mw_1_f9.h"
#include "smp/smp_engine_mw_2.h"
#include "smp/smp_engine_mw_2_f9.h"
#include "smp/smp_engine_mw_3_f8_top.h"
#include "smp/smp_engine_mw_3_f9.h"
#include "utils/utils.h"

CUcontext globalCtx = nullptr;

int main() {
  INIT_GLOG_STR("figure_9");

  const size_t workload_size = 30000;
  // const size_t workload_size = 20000;
  // const size_t workload_size = 12000;
  // const size_t workload_size = 5000;
  // const size_t workload_size = 1000;

  SmpEngineExecutor<bamboosmp::SmpEngineGsF9>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineGs2F9>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF9>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw2F9>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw3F9>(CONGESTED, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineBamboo>(CONGESTED, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineGsF9>(PERFECT, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineGs2F9>(PERFECT, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF9>(PERFECT, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw2F9>(PERFECT, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw3F9>(PERFECT, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineBamboo>(PERFECT, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineGsF9>(RANDOM, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineGs2F9>(RANDOM, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF9>(RANDOM, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw2F9>(RANDOM, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw3F9>(RANDOM, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineBamboo>(RANDOM, workload_size);

  SmpEngineExecutor<bamboosmp::SmpEngineGsF9>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineGs2F9>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMwF9>(SOLO, workload_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw2F9>(SOLO, workload_size);

  // remove SmpEngineMw3F9
  // SmpEngineExecutor<bamboosmp::SmpEngineMw3F9>(SOLO, workload_size);

  SmpEngineExecutorTermination<bamboosmp::SmpEngineBamboo>(SOLO, workload_size);
}
