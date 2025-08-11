#!/usr/bin/env bash
set -euo pipefail

SIZE="${SIZE:-12000}"
GROUP="${GROUP:-10}"

echo "==> Collecting top-of-figure logs (SIZE=$SIZE, GROUP=$GROUP)"

# Solo: CPU only
for e in gs mw la; do
  ./scripts/f7_collect_mem_perf.sh "$e" solo "$SIZE" "$GROUP"
done

# Congested: CPU + GPU
for e in gs mw la; do
  ./scripts/f7_collect_mem_perf.sh "$e" congested "$SIZE" "$GROUP"
done
for e in mw3 la2; do
  ./scripts/f7_collect_mem_ncu.sh "$e" congested "$SIZE" "$GROUP"
done

# Random: CPU + GPU
for e in gs mw la; do
  ./scripts/f7_collect_mem_perf.sh "$e" random "$SIZE" "$GROUP"
done
for e in mw3 la2; do
  ./scripts/f7_collect_mem_ncu.sh "$e" random "$SIZE" "$GROUP"
done

echo "==> Done collecting logs for the top of the figure."
echo "    Files in tmp/logs/"
