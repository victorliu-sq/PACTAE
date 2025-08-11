#!/bin/bash

echo "Executing figure 5 binary..."
./cmake-build-release/benchmark/figure_5

LOGFILE=$(ls -t tmp/logs/figure_5.INFO | head -1)
echo "Processing log: $LOGFILE"

echo "Parsing log file..."
python3 scripts/figure_5_log_parser.py "$LOGFILE" > tmp/metrics/figure_5_parsed_data.json
echo "Generating figure..."
python3 scripts/figure_5_visualize.py tmp/metrics/figure_5_parsed_data.json data/figures/figure_5.png

echo "Workflow completed successfully. Figure saved to data/figures/figure_5.png"