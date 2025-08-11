#!/bin/bash
set -euo pipefail

echo "Executing figure 5 binary..."
# ./cmake-build-release/benchmark/figure_5

LOGFILE=$(ls -t tmp/logs/figure_5.INFO | head -1)
[[ -f "$LOGFILE" ]] || { echo "ERROR: tmp/logs/figure_5.INFO not found"; exit 1; }

mkdir -p tmp/metrics data/figures

echo "Parsing log file..."
python3 scripts/figure_5_log_parser.py "$LOGFILE" > tmp/metrics/figure_5_parsed_data.json

echo "Generating figure..."
python3 scripts/figure_5_visualize.py tmp/metrics/figure_5_parsed_data.json data/figures/figure_5.png

echo "Workflow completed successfully. Figure saved to data/figures/figure_5.png"
