#!/bin/bash
set -euo pipefail

# Step 1: Execute the C++ binary
echo "Table1: Executing the C++ binary..."
./cmake-build-release/benchmark/table_1

# Step 2: Extract core times and store in data/figures
echo "Extracting core times..."
python3 scripts/table_1_extract_core_times.py tmp/logs/table_1.INFO

echo "Done. Core times saved to data/figures/table_1_core_times.txt"
