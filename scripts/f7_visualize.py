#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------- Config -----------------------
MISS_FILE = "tmp/metrics/miss_data.json"
TIME_FILE = "tmp/metrics/timing_data.json"
OUT_FILE = "data/figures/figure_7.png"

hit_color = '#ffffff'
miss_color = '#000000'
highlight_color = '#BB0000'

workloads = ['Solo Case', 'Congested Case', 'Random Case']
miss_keys = {'Solo Case': 'Solo', 'Congested Case': 'Congested', 'Random Case': 'Random'}

implementations_per_workload = {
    'Solo Case': ['GS-Seq-CPU', 'MW-Seq-CPU', 'LA-Seq-CPU'],
    'Congested Case': ['GS-Seq-CPU', 'MW-Seq-CPU', 'LA-Seq-CPU', 'MW-Par-GPU-CAS', 'LA-Par-GPU-CAS'],
    'Random Case': ['GS-Seq-CPU', 'MW-Seq-CPU', 'LA-Seq-CPU', 'MW-Par-GPU-CAS', 'LA-Par-GPU-CAS']
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    miss_data = load_json(MISS_FILE)
    timing_data = load_json(TIME_FILE)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8), dpi=300, sharex=False, sharey='row',
                             gridspec_kw={'height_ratios': [1, 1.2], 'hspace': 0.4})

    # ------------------- Row 1: Cache Misses -------------------
    for i, workload in enumerate(workloads):
        ax = axes[0, i]
        ax.set_facecolor('#eeeeee')
        miss_key = miss_keys[workload]
        impls = implementations_per_workload[workload]
        x = np.arange(len(impls))
        bar_width = 0.7

        miss_vals = []
        hit_vals = []
        miss_percents = []
        for impl in impls:
            key = impl.split('-')[0]
            if 'Par-GPU-CAS' in impl:
                key = 'MWCASKernel' if impl.startswith('MW') else 'LACASKernel'
            total, miss = miss_data[miss_key][key]
            hit = total - miss
            miss_vals.append(miss)
            hit_vals.append(hit)
            miss_percents.append(miss / total * 100 if total else 0)

        ax.bar(x, miss_vals, width=bar_width, color=miss_color, edgecolor='black')
        ax.bar(x, hit_vals, width=bar_width, bottom=miss_vals, color=hit_color, edgecolor='black')

        for j, impl in enumerate(impls):
            color = highlight_color if 'LA' in impl else 'black'
            ax.text(x[j], miss_vals[j] + (0.02 * max(miss_vals)), f'{miss_percents[j]:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(impls, rotation=35, ha='right', fontweight='bold', fontsize=9)
        for label in ax.get_xticklabels():
            if 'LA' in label.get_text():
                label.set_color(highlight_color)

        ax.set_title(workload, fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Cache References", fontsize=10)

    # ------------------- Row 2: Timing -------------------
    cases = workloads
    bar_width = 0.05
    gap = 0.03
    max_algos = max(len(algos) for algos in implementations_per_workload.values())
    x_base = np.arange(max_algos) * (bar_width + gap)

    for i, case in enumerate(cases):
        ax = axes[1, i]
        algos = implementations_per_workload[case]
        init_times = [timing_data[case][algo]["Init Phase"] for algo in algos]
        exec_times = [timing_data[case][algo]["Exec Phase"] for algo in algos]
        x_pos = x_base[:len(algos)]

        ax.bar(x_pos, init_times, bar_width, color='#e9e9e9', hatch='/', edgecolor='black',
               label='Init Phase' if i == 0 else None)
        ax.bar(x_pos, exec_times, bar_width, bottom=init_times, color='#f9f9f9', edgecolor='black',
               label='Exec Phase' if i == 0 else None)

        for j, algo in enumerate(algos):
            init_height = init_times[j]
            total = init_times[j] + exec_times[j]
            color = highlight_color if 'LA' in algo else 'black'
            ax.text(x_pos[j], init_height, f'{init_height:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold' if 'LA' in algo else 'normal', color=color)
            ax.text(x_pos[j], total, f'{total:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold' if 'LA' in algo else 'normal', color=color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(algos, rotation=35, ha='right', fontweight='bold', fontsize=9)
        for label in ax.get_xticklabels():
            if 'LA' in label.get_text():
                label.set_color(highlight_color)

        ax.set_yscale('log')
        ax.set_ylim(0.1, 200)
        if i == 0:
            ax.set_ylabel("Time (seconds)", fontsize=10)

    # ------------------- Final Adjustments -------------------
    plt.subplots_adjust(top=0.92, bottom=0.12, wspace=0.1, left=0.08, right=0.99)

    legend_patches = [
        Patch(facecolor='#000000', edgecolor='black', label='Cache Misses'),
        Patch(facecolor='#ffffff', edgecolor='black', hatch='', label='Cache Hits'),
        Patch(facecolor='#e9e9e9', edgecolor='black', hatch='/', label='Init Phase'),
        Patch(facecolor='#f9f9f9', edgecolor='black', hatch='', label='Exec Phase')
    ]

    fig.legend(handles=legend_patches, loc='upper center', ncol=4, fontsize=10, frameon=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    plt.savefig(OUT_FILE, bbox_inches="tight")
    print(f"Saved {OUT_FILE}")

if __name__ == "__main__":
    main()