#!/usr/bin/env python3
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: figure_9_visualize.py <figure_9.json> <output.png>", file=sys.stderr)
    sys.exit(1)

json_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

ALL_IMPLS = raw["implementations"]
WORKLOADS = raw["workloads"]
D = raw["data"]  # workload -> impl -> dict

# Per requirement: remove MW-Par-GPU-CAS for Solo Case (do not plot that bar)
PLOT_IMPLS_PER_WL = {
    "Perfect Case":   ALL_IMPLS,
    "Solo Case":      [i for i in ALL_IMPLS if i != "MW-Par-GPU-CAS"],
    "Congested Case": ALL_IMPLS,
    "Random Case":    ALL_IMPLS,
}

def get_val(wl, impl, key):
    try:
        return float(D[wl][impl].get(key, 0.0))
    except KeyError:
        return 0.0

# Build stacked components
def build_components(wl, impls):
    pre = [get_val(wl, impl, "precheck") for impl in impls]
    ini = [get_val(wl, impl, "init") for impl in impls]
    exe = [get_val(wl, impl, "core") for impl in impls]  # "core" == execution
    post = [get_val(wl, impl, "post") for impl in impls] # always 0.0 here
    tot = [get_val(wl, impl, "total") for impl in impls]
    return np.array(pre), np.array(ini), np.array(exe), np.array(post), np.array(tot)

# Find best baseline (min total) excluding Bamboo-SMP, and for Solo also excluding MW-Par-GPU-CAS
def best_baseline_index(wl, impls, totals):
    candidates = []
    for idx, impl in enumerate(impls):
        if impl == "Bamboo-SMP":
            continue
        if wl == "Solo Case" and impl == "MW-Par-GPU-CAS":
            continue
        candidates.append((totals[idx], idx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]

fig, axs = plt.subplots(2, 2, figsize=(15, 13), sharey=True, dpi=300)
axs = axs.flatten()

for i, wl in enumerate(WORKLOADS):
    impls = PLOT_IMPLS_PER_WL[wl]
    idx = np.arange(len(impls))
    bar_width = 0.4

    pre, ini, exe, post, tot = build_components(wl, impls)

    axs[i].set_facecolor('#f0f0f0')
    # stacked bars
    p1 = axs[i].bar(idx, pre,  bar_width, label='Prechecking',   hatch='...',  color='#e0e0e0', edgecolor='black')
    p2 = axs[i].bar(idx, ini,  bar_width, bottom=pre,            label='Initialization', color='#ffffff', edgecolor='black')
    p3 = axs[i].bar(idx, exe,  bar_width, bottom=pre+ini,        label='Execution',      hatch='//', color='#b0b0b0', edgecolor='black')
    p4 = axs[i].bar(idx, post, bar_width, bottom=pre+ini+exe,    label='Postprocessing', hatch='--', color='#a0a0a0', edgecolor='black')

    axs[i].set_title(wl, fontweight='bold', fontsize=18)
    axs[i].set_xticks(idx)
    axs[i].set_xticklabels(impls, rotation=45, ha='right', fontweight='bold', fontsize=18)

    # highlight Bamboo on x-axis
    for tick_label in axs[i].get_xticklabels():
        if tick_label.get_text() == 'Bamboo-SMP':
            tick_label.set_color('darkred')

    axs[i].set_yscale('log')
    # set a safe ymin just below min non-zero total
    min_nonzero = np.min(tot[tot > 0]) if np.any(tot > 0) else 0.01
    ymin = max(0.01, min_nonzero / 5.0)
    axs[i].set_ylim(ymin, max(800, float(np.max(tot) * 1.5)))

    # annotate totals + best baseline + Bamboo speedup
    bbi = best_baseline_index(wl, impls, tot)
    best_total = tot[bbi] if bbi is not None else None

    for j, impl in enumerate(impls):
        y = tot[j]
        if y <= 0:
            continue
        # offset so labels don’t overlap the bar
        y_offset = 0.08 if y < 0.2 else 0.12
        text = f"{y:.3f}"
        # mark the best baseline with *
        if bbi is not None and j == bbi:
            text += "$^*$"
        # bamboo speedup
        if impl == "Bamboo-SMP" and best_total and best_total > 0:
            speedup = best_total / y
            text += f" ({speedup:.2f}x$^\\dagger$)"
        color = 'darkred' if impl == 'Bamboo-SMP' else 'black'
        weight = 'bold' if impl == 'Bamboo-SMP' else 'normal'
        axs[i].text(j, y + y_offset, text, ha='center', va='bottom', fontsize=18, color=color, fontweight=weight)

axs[0].set_ylabel('Time (seconds)', fontweight='bold', fontsize=18)
axs[2].set_ylabel('Time (seconds)', fontweight='bold', fontsize=18)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=18)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.subplots_adjust(wspace=0.13)

fig.text(0.02, 0.00, r'$^*$ Best baseline algorithm', fontsize=18, ha='left', va='bottom', fontweight='bold')
fig.text(0.43, 0.00, r'$^\dagger$ Speedup of Bamboo-SMP over the best baseline algorithm', fontsize=18, ha='left', va='bottom', fontweight='bold')

out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"[Fig9] Saved {out_path}")