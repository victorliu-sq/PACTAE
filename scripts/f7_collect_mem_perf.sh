#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/f7_collect_mem_perf.sh <engine: gs|mw|la> <workload: solo|congested|random> <size> [group]
ENGINE="${1:?engine}"; WORKLOAD="${2:?workload}"; SIZE="${3:?size}"; GROUP="${4:-10}"

EXE="${EXE:-./cmake-build-release/benchmark/smp_executor_f7}"
mkdir -p tmp/logs

# Fixed filenames (overwrite each run)
RUN_LOG="tmp/logs/figure_top_run_${ENGINE}_${WORKLOAD}_${SIZE}.log"    # program stdout
PERF_LOG="tmp/logs/figure_top_perf_${ENGINE}_${WORKLOAD}_${SIZE}.log"  # perf stderr

echo "==> perf: $ENGINE $WORKLOAD size=$SIZE group=$GROUP"
perf stat -e cache-references,cache-misses -- \
  "$EXE" --engine "$ENGINE" --workload "$WORKLOAD" --size "$SIZE" --group "$GROUP" \
  1>"$RUN_LOG" 2>"$PERF_LOG"

echo "perf -> $PERF_LOG"
