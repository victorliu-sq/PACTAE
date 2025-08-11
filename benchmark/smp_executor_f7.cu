#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>

#include "smp_engine_exec.h"        // SmpEngineExecutor<>, WorkloadType enum
#include "smp/smp_engine_gs_1_f7.h"
#include "smp/smp_engine_la_f7.h"
#include "smp/smp_engine_mw_1_f7.h"
#include "smp/smp_engine_mw_3_f7.h"
#include "smp/smp_engine_la_2_f7.h"

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog
    << " --engine {gs|mw|la|mw3|la2}"
       " --workload {solo|congested|random}"
       " --size <N>"
       " [--group <G>]\n";
}

static WorkloadType parse_workload(const std::string& s) {
  if (s == "solo") return SOLO;
  if (s == "congested") return CONGESTED;
  if (s == "random") return RANDOM;
  std::cerr << "Unknown workload: " << s << "\n";
  std::exit(2);
}

int main(int argc, char** argv) {
  // Defaults
  std::string engine;
  std::string workload_str;
  size_t size = 0;
  size_t group_size = 10; // only used for RANDOM

  // Very light CLI parsing
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) {
      engine = argv[++i];
    } else if (!std::strcmp(argv[i], "--workload") && i + 1 < argc) {
      workload_str = argv[++i];
    } else if (!std::strcmp(argv[i], "--size") && i + 1 < argc) {
      size = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (!std::strcmp(argv[i], "--group") && i + 1 < argc) {
      group_size = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (!std::strcmp(argv[i], "--help")) {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown/invalid arg: " << argv[i] << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (engine.empty() || workload_str.empty() || size == 0) {
    usage(argv[0]);
    return 2;
  }

  INIT_GLOG_STR("figure_7_single_run");
  WorkloadType workload = parse_workload(workload_str);

  // Always pass group_size; SmpEngineExecutor ignores it for non-RANDOM cases.
  if (engine == "gs") {
    SmpEngineExecutor<bamboosmp::SmpEngineGsF7>(workload, size, group_size);
  } else if (engine == "mw") {
    SmpEngineExecutor<bamboosmp::SmpEngineMwF7>(workload, size, group_size);
  } else if (engine == "la") {
    SmpEngineExecutor<bamboosmp::SmpEngineLaF7>(workload, size, group_size);
  } else if (engine == "mw3") {
    SmpEngineExecutor<bamboosmp::SmpEngineMw3F7>(workload, size, group_size);
  } else if (engine == "la2") {
    SmpEngineExecutor<bamboosmp::SmpEngineLa2F7>(workload, size, group_size);
  } else {
    std::cerr << "Unknown engine: " << engine << "\n";
    SHUTDOWN_GLOG();
    return 2;
  }

  SHUTDOWN_GLOG();
  return 0;
}
