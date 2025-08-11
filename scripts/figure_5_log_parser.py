#!/usr/bin/env python3
import re, sys, json

def norm_engine(tok: str) -> str | None:
    t = tok.strip().replace("_", "-").lower()
    if "single" in t: return "Smp-Init-SingleCore"
    if "multi"  in t: return "Smp-Init-MultiCore"
    if "gpu"    in t or "device" in t: return "Smp-Init-Gpu"
    if "smp-init-singlecore" in t: return "Smp-Init-SingleCore"
    if "smp-init-multicore"  in t: return "Smp-Init-MultiCore"
    if "smp-init-gpu"        in t: return "Smp-Init-Gpu"
    return None

def to_ms(val: float, unit: str) -> float:
    u = unit.lower()
    if u == "ms": return val
    if u in ("us","µs"): return val/1000.0
    if u == "s":  return val*1000.0
    return val

def parse_log(path: str):
    data = {}
    current_key = None

    # Extremely loose “executing” detector (engine + optional workload + size)
    re_exec = re.compile(
        r"Executing.*?(?:SmpInitEngine<\s*([^>]+)\s*>|"
        r"(Smp[-\s]?Init[-\s]?(?:SingleCore|MultiCore|Gpu)|SingleCore|MultiCore|Gpu))"
        r".*?(?:workload\s*[:=]\s*(Solo|Congested|Random))?"
        r".*?(?:size|n|N|SZ|problem[-\s]?size)\s*[:=]\s*(\d+)",
        re.IGNORECASE,
    )

    # Timing lines (lots of variants)
    re_time = re.compile(
        r"(?:(Kernel)(?:\s+Execution)?|"
        r"(Host[-\s]?Device|H2D|HtoD|Memcpy).{0,20}?(Transfer)?|"
        r"(Total\s+Init|Init))"
        r".{0,15}?\bTime\b\s*[:=]?\s*([\d.]+)\s*(us|ms|s)\b",
        re.IGNORECASE,
    )

    # Optional init block marker
    re_block = re.compile(r"Initialize\s+([A-Za-z]+)\s+Profiling\s+Info:", re.IGNORECASE)

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = re_exec.search(line)
            if m:
                raw_engine = (m.group(1) or m.group(2) or "").strip()
                size = int(m.group(4))
                eng = norm_engine(raw_engine) or norm_engine(f"Smp-Init-{raw_engine}")
                if not eng:
                    current_key = None
                    continue
                current_key = f"{eng}_{size}"
                if current_key not in data:
                    data[current_key] = {"engine": eng, "size": size}
                continue

            if current_key:
                b = re_block.search(line)
                if b:
                    data[current_key]["init_type"] = b.group(1)
                    # no continue; timing might be on same line in weird logs

                t = re_time.search(line)
                if t:
                    label_groups = t.groups()
                    # groups: Kernel | HostDevice/H2D/HtoD/Memcpy | 'Transfer'? | Total Init/Init | value | unit
                    kernel, hostlike, _transferword, initlike, val, unit = (
                        label_groups[0],
                        label_groups[1],
                        label_groups[2],
                        label_groups[3],
                        label_groups[4],
                        label_groups[5],
                    )
                    v = to_ms(float(val), unit)

                    if kernel:
                        key = "Kernel"
                    elif initlike:
                        key = "Init"  # includes Total Init
                    elif hostlike:
                        key = "Host-Device Transfer"
                    else:
                        key = "Init"

                    data[current_key][key] = v
    return data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/figure_5_log_parser.py tmp/logs/figure_5.INFO", file=sys.stderr); sys.exit(1)
    result = parse_log(sys.argv[1])
    if not result:
        sys.stderr.write(
            "WARNING: No entries parsed. Run:\n"
            "  grep -nE 'Executing|SmpInitEngine|Init|Kernel|Memcpy|Transfer|size|workload' tmp/logs/figure_5.INFO | sed -n '1,200p'\n"
            "and paste the output so I can key the patterns exactly.\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True))