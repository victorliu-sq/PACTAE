#!/usr/bin/env python3
import re, json, os, sys

LOG = "tmp/logs/figure_7_bottom.INFO"
OUT = "tmp/metrics/timing_data.json"  # keep this path so your next steps don't change

if not os.path.exists(LOG):
    print(f"ERROR: {LOG} not found", file=sys.stderr); sys.exit(1)

# Output skeleton (matches your plotting script)
timing_data = {"Solo Case": {}, "Congested Case": {}, "Random Case": {}}

# Workload label mapping
WL_TITLE = {"Solo": "Solo Case", "Congested": "Congested Case", "Random": "Random Case"}

# Engine label normalization (from raw log header → plot label)
VALID_LABELS = {
    "Gs-Seq-CPU:Figure7":      "GS-Seq-CPU",
    "Mw-Seq-CPU:Figure7":      "MW-Seq-CPU",
    "LA-Seq-Cpu:Figure7":      "LA-Seq-CPU",
    "MW-Par-GPU-CAS:Figure7":  "MW-Par-GPU-CAS",
    "LA-Par-GPU-CAS:Figure7":  "LA-Par-GPU-CAS",
}

# Regexes
exec_re = re.compile(r'Executing\s+(.+?:Figure7)\s+with workload:\s+(Solo|Congested|Random)', re.I)
init_re = re.compile(r'Init Time:\s*([\d.]+)\s*ms')
core_re = re.compile(r'Core Time:\s*([\d.]+)\s*ms')
sep_re  = re.compile(r'^=+$')

# State for the current section
cur_label_raw = None
cur_label = None
cur_wl = None
init_ms = None
core_ms = None

def commit():
    """Commit the current section into timing_data if complete."""
    if cur_label and cur_wl and init_ms is not None and core_ms is not None:
        timing_data[WL_TITLE[cur_wl]][cur_label] = {
            "Init Phase": init_ms / 1000.0,
            "Exec Phase": core_ms / 1000.0,
        }

with open(LOG, "r", errors="ignore") as f:
    for line in f:
        line = line.rstrip("\n")

        m = exec_re.search(line)
        if m:
            # If we were in a previous section, commit it before starting a new one
            commit()
            # Start a new section
            cur_label_raw = m.group(1)
            cur_label = VALID_LABELS.get(cur_label_raw)
            cur_wl = m.group(2)
            init_ms = None
            core_ms = None
            continue

        # If we're inside a section, try to capture times
        if cur_label:
            mi = init_re.search(line)
            if mi:
                init_ms = float(mi.group(1))
                continue

            mc = core_re.search(line)
            if mc:
                core_ms = float(mc.group(1))
                continue

            # Separator means the end of this section
            if sep_re.match(line.strip()):
                commit()
                cur_label_raw = None
                cur_label = None
                cur_wl = None
                init_ms = None
                core_ms = None
                continue

# In case the last section didn't end with a separator
commit()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(timing_data, f, indent=2)

print(f"Wrote {OUT}")
