#include "smp_init_engine_exec.h"
#include "smp/smp_engine_gs_1_f3.h"
#include "smp/smp_engine_mw_1_f3.h"
#include "smp/smp_init_1.h"
#include "smp/smp_init_2.h"
#include "smp/smp_init_3.h"

int main() {
  INIT_GLOG_STR("figure_5");

  // Size 1000
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 1000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 1000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 1000, false);

  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 1000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 1000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 1000, true);

  // Size 5000
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 5000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 5000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 5000, false);

  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 5000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 5000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 5000, true);

  // Size 10000
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 10000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 10000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 10000, false);

  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 10000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 10000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 10000, true);

  // Size 15000
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 15000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 15000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 15000, false);

  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 15000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 15000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 15000, true);

  // Size 30000
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 30000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 30000, false);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 30000, false);

  SmpInitEngineExecutor<bamboosmp::SmpInitEngine1>(CONGESTED, 30000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine2>(CONGESTED, 30000, true);
  SmpInitEngineExecutor<bamboosmp::SmpInitEngine3>(CONGESTED, 30000, true);

  SHUTDOWN_GLOG();
}
