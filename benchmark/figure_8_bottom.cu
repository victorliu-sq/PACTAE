#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>

#include "smp_engine_exec.h"        // SmpEngineExecutor<>, WorkloadType enum, INIT_GLOG_STR, SHUTDOWN_GLOG
#include "smp/smp_engine_la_2_f8_bottom.h"
#include "smp/smp_engine_la_3_f8_bottom.h"
#include "smp/smp_engine_mw_3_f8_bottom.h"
#include "smp/smp_engine_mw_4_f8_bottom.h"

static void usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " --workload {congested|random}"
      " --size <N>"
      " [--group <G>]\n"
      "Notes:\n"
      "  * --group is only used for RANDOM (e.g., 1,5,10,20,30,40,50). Ignored otherwise.\n";
}

static WorkloadType parse_workload(const std::string &s) {
  if (s == "congested") return CONGESTED;
  if (s == "random") return RANDOM;
  std::cerr << "Unknown workload: " << s << "\n";
  std::exit(2);
}

int main(int argc, char **argv) {
  // Required args
  std::string workload_str;
  size_t size = 0;
  size_t group_size = 10; // only used for RANDOM

  // Minimal CLI parsing (same style as your Fig7)
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--workload") && i + 1 < argc) {
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

  if (workload_str.empty() || size == 0) {
    usage(argv[0]);
    return 2;
  }

  INIT_GLOG_STR("figure_8_bottom");
  WorkloadType workload = parse_workload(workload_str);

  // Always pass group_size; executor ignores it for non-RANDOM cases.
  SmpEngineExecutor<bamboosmp::SmpEngineMw3F8Bottom>(workload, size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineMw4F8Bottom>(workload, size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLa2F8Bottom>(workload, size, group_size);
  SmpEngineExecutor<bamboosmp::SmpEngineLa3F8Bottom>(workload, size, group_size);

  SHUTDOWN_GLOG();
  return 0;
}
