import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_data(parsed_data, output_path):
    sizes = [1000, 5000, 10000, 15000, 30000]

    def get_time(engine, size, key):
        task_key = f"{engine}_{size}"
        return parsed_data.get(task_key, {}).get(key, 0.0)

    # CPU bars
    serial_times_rank   = [get_time('Smp-Init-SingleCore', s, 'Init') for s in sizes]
    parallel_times_rank = [get_time('Smp-Init-MultiCore',  s, 'Init') for s in sizes]

    # GPU base = Kernel if present, else Init; top = Host-Device Transfer
    def gpu_base(s):
        k = get_time('Smp-Init-Gpu', s, 'Kernel')
        return k if k > 0 else get_time('Smp-Init-Gpu', s, 'Init')

    gpu_base_rank     = [gpu_base(s) for s in sizes]
    gpu_transfer_rank = [get_time('Smp-Init-Gpu', s, 'Host-Device Transfer') for s in sizes]

    # Reuse for PRMatrix panel
    serial_times_both   = serial_times_rank
    parallel_times_both = parallel_times_rank
    gpu_base_both       = gpu_base_rank
    gpu_transfer_both   = gpu_transfer_rank

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    bar_width = 0.25
    index = np.arange(len(sizes))

    # Panel 1: RankMatrix
    ax1.bar(index,             serial_times_rank,   bar_width, label='Serial CPU',   color='lightgrey')
    ax1.bar(index + bar_width, parallel_times_rank, bar_width, label='Parallel CPU', color='darkgrey', hatch='//')
    ax1.bar(index + 2*bar_width, gpu_base_rank,     bar_width, label='GPU Init',     color='grey', hatch='..')
    ax1.bar(index + 2*bar_width, gpu_transfer_rank, bar_width, bottom=gpu_base_rank, label='GPU Transfer', color='black', hatch='xx')

    ax1.set_xlabel('Sizes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Preprocessing PrefLists into RankMatrix', fontsize=12, fontweight='bold')
    ax1.set_xticks(index + bar_width)
    ax1.set_xticklabels(sizes)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 2e5)

    # Panel 2: PRMatrix (same data buckets in your pipeline)
    ax2.bar(index,             serial_times_both,   bar_width, label='Serial CPU',   color='lightgrey')
    ax2.bar(index + bar_width, parallel_times_both, bar_width, label='Parallel CPU', color='darkgrey', hatch='//')
    ax2.bar(index + 2*bar_width, gpu_base_both,     bar_width, label='GPU Init',     color='grey', hatch='..')
    ax2.bar(index + 2*bar_width, gpu_transfer_both, bar_width, bottom=gpu_base_both, label='GPU Transfer', color='black', hatch='xx')

    ax2.set_xlabel('Sizes', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('Preprocessing PrefLists into PRMatrix', fontsize=12, fontweight='bold')
    ax2.set_xticks(index + bar_width)
    ax2.set_xticklabels(sizes)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-1, 2e5)

    # One shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=12, ncol=2)

    plt.subplots_adjust(top=0.85, wspace=0.4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/figure_5_visualize.py data/parsed/figure_5_parsed_data.json data/figures/figure_5.png")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    plot_data(data, sys.argv[2])
