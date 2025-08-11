#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Optional, Tuple, List, Set

LOGDIR_DEFAULT = "tmp/logs"
OUT_DEFAULT = "tmp/metrics/miss_data.json"

miss_data = {"Solo": {}, "Congested": {}, "Random": {}}
WL_KEY = {"solo": "Solo", "congested": "Congested", "random": "Random"}

CPU_ENGINES = {"gs": "GS", "mw": "MW", "la": "LA"}
GPU_ENGINES = {"mw3": "MWCASKernel", "la2": "LACASKernel"}

def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------------ CPU perf ------------------------------

def parse_perf_counts(text: str):
    refs = miss = None
    for line in text.splitlines():
        if "cache-references" in line:
            tok = line.strip().split()[0]
            num = re.sub(r"[^\d]", "", tok)
            refs = int(num) if num else refs
        elif "cache-misses" in line:
            tok = line.strip().split()[0]
            num = re.sub(r"[^\d]", "", tok)
            miss = int(num) if num else miss
    return refs, miss

# ------------------------------ NCU helpers ------------------------------

def extract_metrics_vals(block: str) -> Optional[Tuple[int,int,int,str]]:
    """Parse a single NCU 'Section: Command line profiler metrics' block."""
    header = None
    for ln in block.splitlines():
        if "Section: Command line profiler metrics" in ln:
            break
        if ln.strip():
            header = ln.strip()

    def last_int(ln: str) -> Optional[int]:
        tok = ln.strip().split()[-1] if ln.strip().split() else ""
        tok = re.sub(r"[^\d]", "", tok)
        return int(tok) if tok else None

    read_sum = hit_sum = miss_sum = None
    for ln in block.splitlines():
        if "lts__t_requests_op_read.sum" in ln:
            read_sum = last_int(ln)
        elif "lts__t_requests_op_read_lookup_hit.sum" in ln:
            hit_sum = last_int(ln)
        elif "lts__t_requests_op_read_lookup_miss.sum" in ln:
            miss_sum = last_int(ln)
    if read_sum is not None and miss_sum is not None:
        return read_sum, (hit_sum or 0), miss_sum, (header or "")
    return None

def split_blocks(text: str) -> List[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    return [b for b in raw_blocks if "Section: Command line profiler metrics" in b]

def best_by_read_sum(blocks: List[str]) -> Optional[Tuple[int,int,int,str]]:
    best = None
    best_r = -1
    for b in blocks:
        parsed = extract_metrics_vals(b)
        if parsed:
            r, *_ = parsed
            if r > best_r:
                best_r = r
                best = parsed
    return best

# ------------------------------ La2 pin (DOTALL regex, last match) ------------------------------

LA2_CORE_RE = re.compile(
    r"La2F7CoreKernel[^\n]*?"                                  # header with kernel name
    r".*?Section:\s*Command line profiler metrics.*?"          # the metrics section
    r"lts__t_requests_op_read\.sum\s+\w+\s+([\d,]+).*?"        # read sum
    r"lookup_hit\.sum\s+\w+\s+([\d,]+).*?"                     # hit sum
    r"lookup_miss\.sum\s+\w+\s+([\d,]+)",                      # miss sum
    re.S  # DOTALL: '.' matches newlines
)

LA2_LEGACY_RE = re.compile(
    r"PlainLACasKernel\([^\n]*?\).*?Section:\s*Command line profiler metrics.*?"
    r"lts__t_requests_op_read\.sum\s+\w+\s+([\d,]+).*?"
    r"lookup_hit\.sum\s+\w+\s+([\d,]+).*?"
    r"lookup_miss\.sum\s+\w+\s+([\d,]+)",
    re.S
)

def parse_la2(text: str) -> Optional[Tuple[int,int]]:
    # Prefer the LAST La2F7CoreKernel section in the file
    last = None
    for m in LA2_CORE_RE.finditer(text):
        last = m
    if last:
        read_sum = int(re.sub(r"[^\d]", "", last.group(1)))
        miss_sum = int(re.sub(r"[^\d]", "", last.group(3)))
        return read_sum, miss_sum

    # Fallback: legacy kernel name (take last)
    last = None
    for m in LA2_LEGACY_RE.finditer(text):
        last = m
    if last:
        read_sum = int(re.sub(r"[^\d]", "", last.group(1)))
        miss_sum = int(re.sub(r"[^\d]", "", last.group(3)))
        return read_sum, miss_sum

    # Last resort: largest read_sum among any metrics blocks
    chosen = best_by_read_sum(split_blocks(text))
    if chosen:
        r, _h, m, _hdr = chosen
        return r, m
    return None

def parse_mw3(text: str) -> Optional[Tuple[int,int]]:
    blocks = split_blocks(text)
    # Prefer KernelWrapperForEach + CoreProc() if available
    cands = [b for b in blocks if "KernelWrapperForEach" in b and "CoreProc()" in b]
    chosen = best_by_read_sum(cands) or best_by_read_sum(blocks)
    if not chosen:
        return None
    r, _h, m, _hdr = chosen
    return r, m

def parse_ncu(text: str, engine: str):
    if not text:
        return None
    if engine == "la2":
        return parse_la2(text)
    if engine == "mw3":
        return parse_mw3(text)
    # generic fallback
    chosen = best_by_read_sum(split_blocks(text))
    if not chosen:
        return None
    r, _h, m, _hdr = chosen
    return r, m

# ------------------------------ Log readers ------------------------------

def add_cpu(engine_short: str, workload: str, size: str, logdir: str):
    alg_key = CPU_ENGINES[engine_short]
    wl_slot = WL_KEY[workload]
    perf_log = os.path.join(logdir, f"figure_top_perf_{engine_short}_{workload}_{size}.log")
    txt = read_text(perf_log)
    if not txt:
        return
    refs, misses = parse_perf_counts(txt)
    if refs is None or misses is None:
        return
    miss_data[wl_slot][alg_key] = (refs, misses)

def add_gpu(engine_short: str, workload: str, size: str, logdir: str):
    kernel_key = GPU_ENGINES[engine_short]
    wl_slot = WL_KEY[workload]
    ncu_log = os.path.join(logdir, f"figure_top_ncu_{engine_short}_{workload}_{size}.log")
    txt = read_text(ncu_log)
    if not txt:
        return
    parsed = parse_ncu(txt, engine_short)
    if not parsed:
        return
    read_sum, miss_sum = parsed
    miss_data[wl_slot][kernel_key] = (read_sum, miss_sum)

# ------------------------------ Size handling ------------------------------

SIZE_PATTERN = re.compile(r".*_(\d+)\.log$")

def discover_sizes(logdir: str) -> Set[str]:
    if not os.path.isdir(logdir):
        return set()
    sizes: Set[str] = set()
    for name in os.listdir(logdir):
        if not name.endswith(".log"):
            continue
        m = SIZE_PATTERN.match(name)
        if m:
            sizes.add(m.group(1))
    return sizes

def resolve_size(cli_size: Optional[str], env_size: Optional[str], logdir: str) -> str:
    if cli_size:
        return cli_size
    if env_size:
        return env_size
    sizes = sorted(discover_sizes(logdir))
    if len(sizes) == 1:
        return sizes[0]
    if not sizes:
        raise SystemExit("No logs found. Please pass --size SIZE or set SIZE env var.")
    raise SystemExit(f"Multiple sizes detected in {logdir}: {', '.join(sizes)}. "
                     f"Please pass --size SIZE (or set SIZE env var).")

# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse Figure 7 TOP memory/cache logs into a JSON metrics file.")
    parser.add_argument("-s", "--size", help="Workload size to select logs for (e.g., 12000).")
    parser.add_argument("--logdir", default=os.environ.get("LOGDIR", LOGDIR_DEFAULT),
                        help=f"Directory containing .log files (default: {LOGDIR_DEFAULT})")
    parser.add_argument("-o", "--out", default=os.environ.get("OUT", OUT_DEFAULT),
                        help=f"Output JSON path (default: {OUT_DEFAULT})")
    args = parser.parse_args()

    size = resolve_size(args.size, os.environ.get("SIZE"), args.logdir)

    # CPU for all workloads
    for wl in ("solo", "congested", "random"):
        for eng in ("gs", "mw", "la"):
            add_cpu(eng, wl, size=size, logdir=args.logdir)

    # GPU for congested + random (NOT solo)
    for wl in ("congested", "random"):
        for eng in ("mw3", "la2"):
            add_gpu(eng, wl, size=size, logdir=args.logdir)

    write_json(args.out, miss_data)
    print(f"Wrote {args.out} (size={size})")

if __name__ == "__main__":
    main()
