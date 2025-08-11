#!/usr/bin/env python3
import os
import re
import sys

LOG_DEFAULT = "tmp/logs/table_1.INFO"
OUT_PATH = "data/figures/table_1_core_times.txt"

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else LOG_DEFAULT
    if not os.path.exists(log_path):
        print(f"[ERR] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    exec_re = re.compile(
        r"Executing\s+(?P<impl>.+?)\s+with workload:\s+(?P<workload>\w+),\s+size:\s+(?P<size>\d+)"
    )
    core_re = re.compile(
        r"\bCore Time:\s+(?P<core_ms>[0-9]+(?:\.[0-9]+)?)\s*ms\b"
    )

    current = {"impl": None, "workload": None, "size": None}
    results = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_exec = exec_re.search(line)
            if m_exec:
                current = {
                    "impl": m_exec.group("impl").strip(),
                    "workload": m_exec.group("workload").strip(),
                    "size": int(m_exec.group("size")),
                }
                continue

            m_core = core_re.search(line)
            if m_core and current["impl"] is not None:
                core_ms = float(m_core.group("core_ms"))
                results.append((
                    current["impl"],
                    current["workload"],
                    current["size"],
                    core_ms
                ))
                current = {"impl": None, "workload": None, "size": None}

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("Implementation\tWorkload\tSize\tCore Time (ms)\n")
        for impl, workload, size, core_ms in results:
            out.write(f"{impl}\t{workload}\t{size}\t{core_ms}\n")

    print(f"[OK] Core times saved to {OUT_PATH}")

if __name__ == "__main__":
    main()