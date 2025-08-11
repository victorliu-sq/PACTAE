#!/bin/bash

echo "Executing figure 9 binary..."
./cmake-build-release/benchmark/figure_9

LOGFILE=$(ls -t tmp/logs/figure_9.INFO | head -1)
echo "Processing log: $LOGFILE"

echo "Parsing log file..."
python3 scripts/figure_9_log_parser.py "$LOGFILE" > tmp/metrics/figure_9.json

echo "Generating figure..."
python3 scripts/figure_9_visualize.py tmp/metrics/figure_9.json data/figures/figure_9.png

echo "Workflow completed. Figure saved at data/figures/figure_9.png"
