#!/usr/bin/env bash
set -euo pipefail

# Usage: workflow/figure_7.sh <SIZE> <GROUP>
SIZE="${1:?usage: $0 SIZE GROUP}"
GROUP="${2:?usage: $0 SIZE GROUP}"

export SIZE
export GROUP

mkdir -p tmp/logs tmp/metrics data/figures

# =================================== Figrue 7 Bottom ======================================================
echo "[F7] Running bottom benchmark…"
# Try with explicit CLI first; if your binary doesn't take args, run without
if ! ./cmake-build-release/benchmark/figure_7_bottom --size "$SIZE" --group "$GROUP"; then
  ./cmake-build-release/benchmark/figure_7_bottom
fi

echo "[F7] Parsing bottom timing…"
# Prefer explicit outputs; if your parser doesn’t support flags, fall back
if ! python3 scripts/f7_parse_bottom_times.py --size "$SIZE" --group "$GROUP" --out tmp/metrics/timing_data.json; then
  python3 scripts/f7_parse_bottom_times.py
fi

# =================================== Figrue 7 Top ======================================================
echo "[F7] Collecting top logs (SIZE=$SIZE, GROUP=$GROUP)…"
bash scripts/f7_collect_top_all.sh


echo "[F7] Parsing top mem/cache…"
python3 scripts/f7_parse_top_mem.py --size "$SIZE" --logdir tmp/logs --out tmp/metrics/miss_data.json

# =================================== Figrue 7 ======================================================
OUT="data/figures/figure_7_${SIZE}_g${GROUP}.png"
echo "[F7] Visualizing → $OUT"
python3 scripts/f7_visualize.py \
  --miss tmp/metrics/miss_data.json \
  --time tmp/metrics/timing_data.json \
  --out "$OUT"

echo "[F7] Done: $OUT"