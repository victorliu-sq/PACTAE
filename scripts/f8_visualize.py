#!/usr/bin/env python3
import json, os, re
import matplotlib
matplotlib.use("Agg")  # don't pop a window
import matplotlib.pyplot as plt

METRICS_DIR = "tmp/metrics"
OUT_DIR = "data/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Load top-row (execution time)
# ----------------------------
with open(os.path.join(METRICS_DIR, "f8_top_times.json"), "r") as f:
    top = json.load(f)

# Sets from top-times (UNCHANGED)
cong_sizes_top = {int(s) for s in top.get("congested", {}).keys()}
rand_sizes_top = {int(s) for s in top.get("random", {}).keys()}
random_size = sorted(rand_sizes_top)[0] if rand_sizes_top else 30000
rand_groups_top = {int(g) for g in top.get("random", {}).get(str(random_size), {}).keys()}

# Helpers to fetch times from top-times
def times_congested(engine, sizes):
    return [top["congested"][str(s)].get(engine) for s in sizes]

def times_random(engine, size, groups):
    return [top["random"][str(size)][str(g)].get(engine) for g in groups]

# -----------------------------------
# Load bottom-row (atomic op counters)
# -----------------------------------
# Map NEW kernel names -> display labels used in the figure (same as top-times keys)
KERNEL_MAP = {
    "Mw3F8CASBottomKernel": "MW-Par-GPU-CAS",
    "Mw4F8BottomKernel":    "MW-Par-GPU-MIN",
    "La2F8CASBottomKernel": "LA-Par-GPU-CAS",
    "La3MinF8BottomKernel": "LA-Par-GPU-MIN",
}

# Congested atoms (from files)
cong_atomic_files = {}
for fname in os.listdir(METRICS_DIR):
    m = re.match(r"f8_congested_(\d+)\.json$", fname)
    if not m:
        continue
    size = int(m.group(1))
    with open(os.path.join(METRICS_DIR, fname), "r") as f:
        data = json.load(f)
    mapped = {}
    for k, v in data.items():
        if k in KERNEL_MAP:
            mapped[KERNEL_MAP[k]] = int(v)
    cong_atomic_files[size] = mapped

# Random atoms (from files)
rand_atomic_files = {}
for fname in os.listdir(METRICS_DIR):
    m = re.match(r"f8_random_(\d+)_g(\d+)\.json$", fname)
    if not m:
        continue
    size = int(m.group(1)); group = int(m.group(2))
    if size != random_size:  # align to top-times size
        continue
    with open(os.path.join(METRICS_DIR, fname), "r") as f:
        data = json.load(f)
    mapped = {}
    for k, v in data.items():
        if k in KERNEL_MAP:
            mapped[KERNEL_MAP[k]] = int(v)
    rand_atomic_files[group] = mapped

# ----------------------------
# Align axes (intersection)
# ----------------------------
cong_sizes_common = sorted(cong_sizes_top & set(cong_atomic_files.keys()))
rand_groups_common = sorted(rand_groups_top & set(rand_atomic_files.keys()))

# Convenience getters for atomics (robust to missing labels)
def atoms_congested(label):
    return [cong_atomic_files.get(s, {}).get(label) for s in cong_sizes_common]

def atoms_random(label):
    return [rand_atomic_files.get(g, {}).get(label) for g in rand_groups_common]

# Build time series aligned to the same x
mw_cas_times_c = times_congested("MW-Par-GPU-CAS", cong_sizes_common)
mw_min_times_c = times_congested("MW-Par-GPU-MIN", cong_sizes_common)
la_cas_times_c = times_congested("LA-Par-GPU-CAS",  cong_sizes_common)
la_min_times_c = times_congested("LA-Par-GPU-MIN",  cong_sizes_common)

mw_cas_times_r = times_random("MW-Par-GPU-CAS", random_size, rand_groups_common)
mw_min_times_r = times_random("MW-Par-GPU-MIN", random_size, rand_groups_common)
la_cas_times_r = times_random("LA-Par-GPU-CAS",  random_size, rand_groups_common)
la_min_times_r = times_random("LA-Par-GPU-MIN",  random_size, rand_groups_common)

# ------------------------------------------------------
# Add synthetic "group = 1" column on the right panels
# -> reuse the LAST column of the *left* neighbor.
#    (largest congested size's values)
# ------------------------------------------------------
if cong_sizes_common:
    max_cong_size = cong_sizes_common[-1]  # last column of left neighbor

    # Execution times from congested[max_size]
    cong_last = top["congested"].get(str(max_cong_size), {})
    synth_times = {
        "MW-Par-GPU-CAS": cong_last.get("MW-Par-GPU-CAS"),
        "MW-Par-GPU-MIN": cong_last.get("MW-Par-GPU-MIN"),
        "LA-Par-GPU-CAS": cong_last.get("LA-Par-GPU-CAS"),
        "LA-Par-GPU-MIN": cong_last.get("LA-Par-GPU-MIN"),
    }

    # Prepend group=1
    rand_groups_aug = [1] + rand_groups_common
    mw_cas_times_r = [synth_times["MW-Par-GPU-CAS"]] + mw_cas_times_r
    mw_min_times_r = [synth_times["MW-Par-GPU-MIN"]] + mw_min_times_r
    la_cas_times_r = [synth_times["LA-Par-GPU-CAS"]] + la_cas_times_r
    la_min_times_r = [synth_times["LA-Par-GPU-MIN"]] + la_min_times_r

    # Atomics: reuse congested[max_size] per label
    def atoms_random_aug(label):
        left_last = cong_atomic_files.get(max_cong_size, {}).get(label)
        return [left_last] + atoms_random(label)
else:
    # fallback: no congested -> don't synthesize
    rand_groups_aug = rand_groups_common
    atoms_random_aug = atoms_random

# -----------------------------------
# Plot (same color/marker style)
# -----------------------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=300)

axes[0][0].set_title('Congested Case', fontweight='bold', fontsize=12)
axes[0][1].set_title('Random Case', fontweight='bold', fontsize=12)
for ax in axes[1]:
    ax.set_facecolor('#f0f0f0')

# Colors/markers: MW-CAS black 'o'; MW-MIN #BB0000 's'; LA-CAS gray '^'; LA-MIN #4B0082 'x'

# Top-left: execution times (congested)
axes[0][0].plot(cong_sizes_common, mw_cas_times_c, label='MW-Par-GPU-CAS', marker='o', color='black')
axes[0][0].plot(cong_sizes_common, mw_min_times_c, label='MW-Par-GPU-MIN', marker='s', color='#BB0000')
axes[0][0].plot(cong_sizes_common, la_cas_times_c, label='LA-Par-GPU-CAS', marker='^', color='gray')
axes[0][0].plot(cong_sizes_common, la_min_times_c, label='LA-Par-GPU-MIN', marker='x', color='#4B0082')
axes[0][0].set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=10)
axes[0][0].legend(loc='upper left', fontsize=8)
axes[0][0].grid(True, linestyle='--')

# Top-right: execution times (random)  (includes synthetic group=1)
axes[0][1].plot(rand_groups_aug, mw_cas_times_r, label='MW-Par-GPU-CAS', marker='o', color='black')
axes[0][1].plot(rand_groups_aug, mw_min_times_r, label='MW-Par-GPU-MIN', marker='s', color='#BB0000')
axes[0][1].plot(rand_groups_aug, la_cas_times_r, label='LA-Par-GPU-CAS', marker='^', color='gray')
axes[0][1].plot(rand_groups_aug, la_min_times_r, label='LA-Par-GPU-MIN', marker='x', color='#4B0082')
axes[0][1].set_xlabel('Group Size', fontweight='bold', fontsize=10)
axes[0][1].legend(loc='upper right', fontsize=8)
axes[0][1].grid(True, linestyle='--')

# Bottom-left: atomic reads (congested)
axes[1][0].plot(cong_sizes_common, atoms_congested('MW-Par-GPU-CAS'), label='MW-Par-GPU-CAS', marker='o', color='black')
axes[1][0].plot(cong_sizes_common, atoms_congested('MW-Par-GPU-MIN'), label='MW-Par-GPU-MIN', marker='s', color='#BB0000')
axes[1][0].plot(cong_sizes_common, atoms_congested('LA-Par-GPU-CAS'),  label='LA-Par-GPU-CAS',  marker='^', color='gray')
axes[1][0].plot(cong_sizes_common, atoms_congested('LA-Par-GPU-MIN'), label='LA-Par-GPU-MIN', marker='x', color='#4B0082')
axes[1][0].set_xlabel('Workload Size', fontweight='bold', fontsize=10)
axes[1][0].set_ylabel('#Atomic Reads (sectors)', fontweight='bold', fontsize=10)
axes[1][0].legend(loc='upper left', fontsize=8)
axes[1][0].grid(True, linestyle='--')

# Bottom-right: atomic reads (random)  (includes synthetic group=1)
axes[1][1].plot(rand_groups_aug, atoms_random_aug('MW-Par-GPU-CAS'), label='MW-Par-GPU-CAS', marker='o', color='black')
axes[1][1].plot(rand_groups_aug, atoms_random_aug('MW-Par-GPU-MIN'), label='MW-Par-GPU-MIN', marker='s', color='#BB0000')
axes[1][1].plot(rand_groups_aug, atoms_random_aug('LA-Par-GPU-CAS'),  label='LA-Par-GPU-CAS',  marker='^', color='gray')
axes[1][1].plot(rand_groups_aug, atoms_random_aug('LA-Par-GPU-MIN'), label='LA-Par-GPU-MIN', marker='x', color='#4B0082')
axes[1][1].set_xlabel('Group Size', fontweight='bold', fontsize=10)
axes[1][1].legend(loc='upper right', fontsize=8)
axes[1][1].grid(True, linestyle='--')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "figure_8.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"[OK] Saved {out_path}")
