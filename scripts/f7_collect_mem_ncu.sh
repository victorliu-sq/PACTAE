#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/f7_collect_mem_ncu.sh <engine: mw3|la2> <workload: solo|congested|random> <size> [group]
ENGINE="${1:?engine}"; WORKLOAD="${2:?workload}"; SIZE="${3:?size}"; GROUP="${4:-10}"

EXE="${EXE:-./cmake-build-release/benchmark/smp_executor_f7}"
EXE="$(realpath -m "$EXE")"
if [[ ! -x "$EXE" ]]; then
  echo "ERROR: Executable not found or not executable: $EXE" >&2
  exit 1
fi

mkdir -p tmp/logs
NCU_LOG="tmp/logs/figure_top_ncu_${ENGINE}_${WORKLOAD}_${SIZE}.log"

# Keep metrics as a single argument
METRICS="lts__t_requests_op_read.sum,lts__t_requests_op_read_lookup_hit.sum,lts__t_requests_op_read_lookup_miss.sum"

echo "==> ncu: $ENGINE $WORKLOAD size=$SIZE group=$GROUP"
echo "    EXE: $EXE"
echo "    LOG: $NCU_LOG"

# IMPORTANT: do NOT use the `--` separator with your ncu version
# Exact style as your working command
if ! ncu --metrics "$METRICS" \
     "$EXE" --engine "$ENGINE" --workload "$WORKLOAD" --size "$SIZE" --group "$GROUP" \
     >"$NCU_LOG" 2>&1; then
  echo "!! ncu failed. Tail of $NCU_LOG:" >&2
  tail -n 40 "$NCU_LOG" >&2 || true
  exit 1
fi

echo "ncu -> $NCU_LOG"