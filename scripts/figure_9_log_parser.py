#!/usr/bin/env python3
"""
Robust Figure 9 log -> JSON converter (Python 3.9 compatible).

- Captures the whole engine name up to ' with workload:' (handles '+' in names).
- Reads only the timing lines inside the 'Profiling Info:' block until the '====' rule.
- Works with optional ', group_size: N'.
- Outputs seconds.
"""

import sys, re, json
from collections import defaultdict
from typing import Optional

if len(sys.argv) != 2:
    print("Usage: figure_9_log_parser_alt.py <path/to/figure_9.INFO>", file=sys.stderr)
    sys.exit(1)

log_path = sys.argv[1]

def ms_to_s(v: str) -> float:
    return float(v) / 1000.0

def normalize_impl(raw_name: str) -> Optional[str]:
    n = raw_name.lower()
    if "gs-seq-cpu" in n:      return "GS-Seq-CPU"
    if "gs-par-cpu" in n:      return "GS-Par-CPU"
    if "mw-seq-cpu" in n:      return "MW-Seq-CPU"
    if "mw-par-cpu" in n:      return "MW-Par-CPU"
    if "mw-par-gpu-cas" in n:  return "MW-Par-GPU-CAS"
    if "bamboosmp" in n:       return "Bamboo-SMP"
    return None

WL_MAP = {
    "Perfect":   "Perfect Case",
    "Solo":      "Solo Case",
    "Congested": "Congested Case",
    "Random":    "Random Case",
}

RE_EXEC = re.compile(
    r"\] Executing (?P<name>.+?) with workload: (?P<wl>Perfect|Solo|Congested|Random)\b",
    re.IGNORECASE,
)
RE_PROF_HEADER = re.compile(r"\] .*Profiling Info:\s*$")
RE_BLOCK_END   = re.compile(r"\] =+")
RE_PRE   = re.compile(r"\] .* Precheck Time: (?P<v>[0-9.]+) ms")
RE_INIT  = re.compile(r"\] .* Init Time: (?P<v>[0-9.]+) ms")
RE_CORE  = re.compile(r"\] .* Core Time: (?P<v>[0-9.]+) ms")
RE_TOTAL = re.compile(r"\] .* Total Time: (?P<v>[0-9.]+) ms")

data = defaultdict(lambda: defaultdict(lambda: dict(
    precheck=0.0, init=0.0, core=0.0, post=0.0, total=0.0
)))

cur_wl = None
cur_impl = None
in_prof = False

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        m = RE_EXEC.search(line)
        if m:
            raw_name = m.group("name").strip()
            wl_key   = m.group("wl")
            wl = WL_MAP.get(wl_key, wl_key)
            impl = normalize_impl(raw_name)
            cur_wl, cur_impl, in_prof = wl, impl, False
            continue

        if cur_wl and cur_impl and RE_PROF_HEADER.search(line):
            in_prof = True
            continue

        if in_prof and cur_wl and cur_impl:
            if RE_BLOCK_END.search(line):
                in_prof = False
                continue
            m = RE_PRE.search(line)
            if m:
                data[cur_wl][cur_impl]["precheck"] = ms_to_s(m.group("v")); continue
            m = RE_INIT.search(line)
            if m:
                data[cur_wl][cur_impl]["init"] = ms_to_s(m.group("v")); continue
            m = RE_CORE.search(line)
            if m:
                data[cur_wl][cur_impl]["core"] = ms_to_s(m.group("v")); continue
            m = RE_TOTAL.search(line)
            if m:
                data[cur_wl][cur_impl]["total"] = ms_to_s(m.group("v")); continue

out = {
    "implementations": ["GS-Seq-CPU","GS-Par-CPU","MW-Seq-CPU","MW-Par-CPU","MW-Par-GPU-CAS","Bamboo-SMP"],
    "workloads": ["Perfect Case","Solo Case","Congested Case","Random Case"],
    "data": data,
}

print(json.dumps(out, indent=2, sort_keys=True))