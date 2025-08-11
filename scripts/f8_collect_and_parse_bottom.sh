#!/usr/bin/env bash
set -euo pipefail

# Config
EXE="./cmake-build-release/benchmark/figure_8_bottom"
LOG_DIR="tmp/logs"
METRIC_LIST="l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom.sum,l1tex__m_xbar2l1tex_read_sectors_mem_global_op_atom.sum"
#METRIC_LIST="l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum"
#METRIC_LIST="lts__t_requests_op_atom.sum,lts__t_sectors_op_atom.sum,lts__t_requests_op_red.sum,lts__t_sectors_op_red.sum"

mkdir -p "$LOG_DIR"

# --- Congested sizes ---
CONG_SIZES=(5000 10000 15000 20000 25000 30000)
#CONG_SIZES=(5000 10000 15000)

for SZ in "${CONG_SIZES[@]}"; do
  LOG="${LOG_DIR}/figure_8_congested_${SZ}.log"
  echo "==> ncu (congested, size=${SZ}) -> ${LOG}"
  ncu --metrics "${METRIC_LIST}" \
    --log-file "${LOG}" \
    "${EXE}" --workload congested --size "${SZ}"

  echo "==> parse -> tmp/metrics/f8_congested_${SZ}.json"
  python3 scripts/f8_parse_bottom.py congested "${SZ}"
done

# --- Random size + groups ---
RAND_SIZE=30000
#RAND_SIZE=5000
RGROUPS=(5 10 20 30 40 50)

for G in "${RGROUPS[@]}"; do
  LOG="${LOG_DIR}/figure_8_random_${RAND_SIZE}_g${G}.log"
  echo "==> ncu (random, size=${RAND_SIZE}, group=${G}) -> ${LOG}"
  ncu --metrics "${METRIC_LIST}" \
    --log-file "${LOG}" \
    "${EXE}" --workload random --size "${RAND_SIZE}" --group "${G}"

  echo "==> parse -> tmp/metrics/f9_random_${RAND_SIZE}_g${G}.json"
  python3 scripts/f8_parse_bottom.py random "${RAND_SIZE}" "${G}"
done

echo "[OK] Completed all collections and parses.]"
