#!/usr/bin/env python3
import sys
import os
import re
import json

def usage():
    print(f"Usage: {sys.argv[0]} <workload> <size> [group]")
    print("Examples:")
    print("  python3 scripts/f8_parse_bottom.py congested 15000")
    print("  python3 scripts/f8_parse_bottom.py random 15000 10   # parses *_g10.log")
    sys.exit(2)

if len(sys.argv) not in (3, 4):
    usage()

workload = sys.argv[1].lower()
size = sys.argv[2]
group = sys.argv[3] if len(sys.argv) == 4 else None

# Build log path (same as before)
log_path = f"tmp/logs/figure_8_{workload}_{size}"
if group is not None:
    log_path += f"_g{group}"
log_path += ".log"

# Build output path (same as before)
out_dir = "tmp/metrics"
out_path = f"{out_dir}/f8_{workload}_{size}"
if group is not None:
    out_path += f"_g{group}"
out_path += ".json"

if not os.path.exists(log_path):
    print(f"[ERR] Log file not found: {log_path}")
    sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

# --- Kernel name handling ----------------------------------------------------
# We'll accept the four kernel names in either the "==PROF== Profiling" header
# or in the function-signature line that starts the metric table.
KERNEL_LABELS = {
    "Mw3F8CASBottomKernel": "Mw3F8CASBottomKernel",
    "Mw4F8BottomKernel": "Mw4F8BottomKernel",
    "La2F8CASBottomKernel": "La2F8CASBottomKernel",
    "La3MinF8BottomKernel": "La3MinF8BottomKernel",
}

# Match: ==PROF== Profiling "NAME" - ...
profiling_hdr_re = re.compile(r'^==PROF==\s+Profiling\s+"([^"]+)"')

# Match: function-signature line that begins a kernel report block, e.g.:
#   Mw3F8CASBottomKernel(int, const int *, ...) (40, 1, 1)x(128, 1, 1), ...
func_sig_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s*\(')

# The metric we want; number may have commas
metric_write_re = re.compile(
    r"l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom\.sum\s+sector\s+([0-9,]+)\s*$"
)

results = {}
current_kernel = None  # will hold a canonical kernel key (one of KERNEL_LABELS values)

def normalize_kernel(name: str):
    """Return the exact key if 'name' is one of the known kernels; else None."""
    return KERNEL_LABELS.get(name)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for raw in f:
        line = raw.rstrip("\n")

        # 1) Try to catch kernel from "==PROF==" header
        m_prof = profiling_hdr_re.search(line)
        if m_prof:
            kname = m_prof.group(1)
            nk = normalize_kernel(kname)
            current_kernel = nk if nk else None
            continue

        # 2) Try to catch kernel from the function-signature line
        m_func = func_sig_re.search(line)
        if m_func:
            kname = m_func.group(1)
            nk = normalize_kernel(kname)
            # Only set/override current_kernel if this is one we care about
            if nk:
                current_kernel = nk
            else:
                # entering some other function block; stop tracking
                current_kernel = None
            continue

        # 3) If we're inside one of the four kernels, read the metric
        if current_kernel:
            m_metric = metric_write_re.search(line)
            if m_metric:
                val = int(m_metric.group(1).replace(",", ""))
                if val != 0:
                    results[current_kernel] = val
                # Done with this kernel report block; clear until we see the next kernel
                current_kernel = None

# Save JSON
with open(out_path, "w") as fo:
    json.dump(results, fo, indent=2)

print(f"[OK] Parsed metrics saved to {out_path}")
print(json.dumps(results, indent=2))