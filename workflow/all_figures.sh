#!/usr/bin/env bash
set -euo pipefail

# Root of project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== BambooSMPAE: Running all figure workflows ==="
echo "Root directory: ${ROOT_DIR}"
echo

# Ensure output directories exist
mkdir -p "${ROOT_DIR}/tmp/logs" \
         "${ROOT_DIR}/tmp/metrics" \
         "${ROOT_DIR}/data/figures" \
         "${ROOT_DIR}/data/parsed"


#run_workflow() {
#    local script="$1"
#    echo "------------------------------------------------------------"
#    echo ">>> Running: ${script}"
#    echo "------------------------------------------------------------"
#    bash "${ROOT_DIR}/${script}"
#    echo
#}

# replace your run_workflow() with this:
run_workflow() {
    local script="$1"; shift
    echo "------------------------------------------------------------"
    echo ">>> Running: ${script} $*"
    echo "------------------------------------------------------------"
    bash "${ROOT_DIR}/${script}" "$@"
    echo
}

# List of workflows to run in order
WORKFLOWS=(
    "workflow/figure_3.sh"
    "workflow/table_1.sh"
    "workflow/figure_5.sh"
    "workflow/figure_7.sh"    # requires SIZE and GROUP
    "workflow/figure_8.sh"
    "workflow/figure_9.sh"
)

# Parameters for Figure 7 (you can adjust)
FIG7_SIZE=20000
FIG7_GROUP=5

# Loop through and run workflows
for wf in "${WORKFLOWS[@]}"; do
    if [[ "$wf" == "workflow/figure_7.sh" ]]; then
        run_workflow "$wf" "$FIG7_SIZE" "$FIG7_GROUP"
    else
        run_workflow "$wf"
    fi
done

echo "✅ All workflows completed successfully."
echo "Figures and outputs are available in: ${ROOT_DIR}/data/figures"