#!/usr/bin/env bash
set -euo pipefail

# Paths
LOG="tmp/logs/figure_3.INFO"
PARSED="tmp/metrics/figure_3.json"
FIG="data/figures/figure_3.png"

# Ensure dirs
mkdir -p tmp/logs tmp/metrics data/figures

echo "Figure3: Executing the C++ binary..."
./cmake-build-release/benchmark/figure_3

echo "Processing log: $LOG"
if [[ ! -f "$LOG" ]]; then
  echo "ERROR: Log not found: $LOG"
  exit 1
fi

echo "Parsing log..."
python3 scripts/figure_3_log_parser.py "$LOG" > "$PARSED"

# Optional: quick sanity check to fail early if parsing missed keys
python3 - <<'PY'
import json, sys
p="tmp/metrics/figure_3.json"
d=json.load(open(p))
for algo in ("Gs-Seq-CPU-Profile","Mw-Seq-CPU-Profile"):
    assert algo in d, f"Missing {algo} in {p}"
    for wl in ("Solo","Random","Congested"):
        assert wl in d[algo], f"Missing workload {wl} for {algo}"
        for k in ("Core","Random Access"):
            assert k in d[algo][wl], f"Missing '{k}' for {algo}/{wl}"
print("Parsed data OK.")
PY

echo "Generating and saving figure..."
python3 scripts/figure_3_visualize.py "$PARSED" "$FIG"

echo "Workflow completed. Figure saved to $FIG"
