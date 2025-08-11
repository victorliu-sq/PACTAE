#!/usr/bin/env python3
import os
import re
import json
import argparse
from collections import defaultdict

# --- CLI ---
parser = argparse.ArgumentParser(
    description="Parse Figure 8 (top) execution times from split logs and merge to one JSON."
)
parser.add_argument(
    "--left",
    default="tmp/logs/figure_8_top_left.INFO",
    help="Path to the congested (left) log file.",
)
parser.add_argument(
    "--right",
    default="tmp/logs/figure_8_top_right.INFO",
    help="Path to the random (right) log file.",
)
parser.add_argument(
    "--out",
    default="tmp/metrics/f8_top_times.json",
    help="Output JSON path (merged).",
)
args = parser.parse_args()

# Ensure output dir exists
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# --- Data containers (same shape as before) ---
# {
#   "congested": { "5000": {"MW-Par-GPU-CAS": 211.015, ...}, "10000": {...}, ... },
#   "random": { "30000": { "5": {"MW-Par-GPU-CAS": 847.437, ...}, "10": {...}, ... } }
# }
congested = defaultdict(dict)
random = defaultdict(lambda: defaultdict(dict))

# --- Regexes ---
# Accepts:
#   Executing MW-Par-GPU-CAS with workload: Congested, size: 5000
#   Executing MW-Par-GPU-MIN:Figure8-TLeft with workload: Congested, size: 5000
#   Executing MW-Par-GPU-CAS with workload: Random, size: 30000, group_size: 5
#   Executing LA-Par-GPU-MIN:Figure8-TRight with workload: Random, size: 30000, group_size: 50
exec_re = re.compile(
    r"Executing\s+"
    r"(MW-Par-GPU-(?:CAS|MIN)|LA-Par-GPU-(?:CAS|MIN))"  # engine
    r"(?::[A-Za-z0-9\-]+)?"                              # optional suffix (:Figure8, :Figure8-TLeft, :Figure8-TRight, etc.)
    r"\s+with workload:\s+"
    r"(Congested|Random),\s+size:\s+(\d+)"               # workload, size
    r"(?:,\s*group_size:\s*(\d+))?",                     # optional group_size
    re.IGNORECASE,
)

# Prefer Total Time; fall back to Core Time if Total is absent
total_re = re.compile(r"Total Time:\s*([\d.]+)\s*ms", re.IGNORECASE)
core_re  = re.compile(r"Core Time:\s*([\d.]+)\s*ms",  re.IGNORECASE)

def parse_one_file(path: str):
    """Return list of entries parsed from a single INFO log."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        m = exec_re.search(lines[i])
        if not m:
            i += 1
            continue

        engine   = m.group(1)                 # e.g., MW-Par-GPU-CAS
        workload = m.group(2).lower()         # 'congested' | 'random'
        size     = m.group(3)                 # '5000', '30000', ...
        group    = m.group(4)                 # '5', '10', ..., or None

        # Look ahead a few lines for timing; Total preferred, else Core
        time_ms = None
        j_limit = min(i + 20, len(lines))
        j = i + 1
        while j < j_limit:
            mt = total_re.search(lines[j])
            if mt:
                time_ms = float(mt.group(1))
                break
            mc = core_re.search(lines[j])
            if mc:
                time_ms = float(mc.group(1))
                break
            j += 1

        if time_ms is not None:
            entries.append(dict(
                engine=engine,
                workload=workload,
                size=size,
                group=group,
                time_ms=time_ms,
            ))
        i = j  # skip past what we consumed
        i += 1

    return entries

# --- Parse both files and merge ---
all_entries = []
for label, path in (("left", args.left), ("right", args.right)):
    try:
        parsed = parse_one_file(path)
        all_entries.extend(parsed)
        print(f"[OK] Parsed {len(parsed):>3} entries from {label}: {path}")
    except FileNotFoundError:
        print(f"[WARN] Missing {label} log: {path}")

# Fill dicts
for e in all_entries:
    if e["workload"] == "congested":
        congested[e["size"]][e["engine"]] = e["time_ms"]
    else:  # random
        if e["group"] is not None:  # only store when group_size is present
            random[e["size"]][e["group"]][e["engine"]] = e["time_ms"]

# Sort keys numerically and dump
result = {
    "congested": {
        size: dict(vals)
        for size, vals in sorted(congested.items(), key=lambda kv: int(kv[0]))
    },
    "random": {
        size: {
            grp: dict(vals)
            for grp, vals in sorted(groups.items(), key=lambda kv: int(kv[0]))
        }
        for size, groups in sorted(random.items(), key=lambda kv: int(kv[0]))
    },
}

with open(args.out, "w") as fo:
    json.dump(result, fo, indent=2)

# Summary
c_sizes = ", ".join(sorted(result["congested"].keys(), key=int)) or "none"
r_sizes = ", ".join(sorted(result["random"].keys(), key=int)) or "none"
print(f"[DONE] Wrote {args.out}")
print(f"        Congested sizes: {c_sizes}")
print(f"        Random sizes:    {r_sizes}")