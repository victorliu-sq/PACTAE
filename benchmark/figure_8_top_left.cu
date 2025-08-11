#include "smp_engine_exec.h"
#include "smp_init_engine_exec.h"
#include "smp/smp_engine_la_2_f8_top.h"
#include "smp/smp_engine_la_3_f8_tleft.h"
#include <smp/smp_engine_mw_3_f8_top.h>
#include "smp/smp_engine_mw_4_f8_tleft.h"

int main() {
  INIT_GLOG_STR("figure_8_top_left");

  // ========================================================================================
  // Congested Case
  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 5000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 5000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 5000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 5000);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 10000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 10000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 10000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 10000);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 15000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 15000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 15000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 15000);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 20000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 20000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 20000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 20000);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 25000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 25000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 25000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 25000);

  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Top>(CONGESTED, 30000);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8TLeft>(CONGESTED, 30000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Top>(CONGESTED, 30000);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8TLeft>(CONGESTED, 30000);

  SHUTDOWN_GLOG();
}
