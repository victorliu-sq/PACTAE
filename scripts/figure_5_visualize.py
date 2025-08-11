import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_data(parsed_data, output_path):
    sizes = [1000, 5000, 10000, 15000, 30000]

    # Helper function
    def get_time(engine, size, key):
        task_key = f"{engine}_{size}"
        return parsed_data.get(task_key, {}).get(key, 0)

    serial_times_rank = [get_time('Smp-Init-SingleCore', s, 'Init') for s in sizes]
    parallel_cpu_times_rank = [get_time('Smp-Init-MultiCore', s, 'Init') for s in sizes]
    gpu_kernel_rank = [get_time('Smp-Init-Gpu', s, 'Kernel') for s in sizes]
    gpu_transfer_rank = [get_time('Smp-Init-Gpu', s, 'Host-Device Transfer') for s in sizes]

    # For PRMatrix + RankMatrix ("Both"), we will reuse the same data as you provided
    serial_times_both = serial_times_rank
    parallel_cpu_times_both = parallel_cpu_times_rank
    gpu_kernel_both = gpu_kernel_rank
    gpu_transfer_both = gpu_transfer_rank

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    bar_width = 0.25
    index = np.arange(len(sizes))

    # Plot RankMatrix
    ax1.bar(index, serial_times_rank, bar_width, label='Serial CPU', color='lightgrey')
    ax1.bar(index + bar_width, parallel_cpu_times_rank, bar_width, label='Parallel CPU', color='darkgrey', hatch='//')
    ax1.bar(index + 2*bar_width, gpu_kernel_rank, bar_width, label='GPU Kernel', color='grey', hatch='..')
    ax1.bar(index + 2*bar_width, gpu_transfer_rank, bar_width, bottom=gpu_kernel_rank, label='GPU Transfer', color='black', hatch='xx')

    ax1.set_xlabel('Sizes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax1.set_title('Preprocessing PrefLists into RankMatrix', fontsize=12, fontweight='bold')
    ax1.set_xticks(index + bar_width)
    ax1.set_xticklabels(sizes)
    ax1.set_yscale('log')
    ax1.set_ylim(10**-1, 10**5 * 2)

    # Plot PRMatrix (Both)
    ax2.bar(index, serial_times_both, bar_width, label='Serial CPU', color='lightgrey')
    ax2.bar(index + bar_width, parallel_cpu_times_both, bar_width, label='Parallel CPU', color='darkgrey', hatch='//')
    ax2.bar(index + 2*bar_width, gpu_kernel_both, bar_width, label='GPU Kernel', color='grey', hatch='..')
    ax2.bar(index + 2*bar_width, gpu_transfer_both, bar_width, bottom=gpu_kernel_both, label='GPU Transfer', color='black', hatch='xx')

    ax2.set_xlabel('Sizes', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax2.set_title('Preprocessing PrefLists into PRMatrix', fontsize=12, fontweight='bold')
    ax2.set_xticks(index + bar_width)
    ax2.set_xticklabels(sizes)
    ax2.set_yscale('log')
    ax2.set_ylim(10**-1, 10**5 * 2)

    # Single shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=12, ncol=2)

    plt.subplots_adjust(top=0.85, wspace=0.4)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/figure_5_visualize.py data/parsed/figure_5_parsed_data.json data/figures/figure_5.png")
        sys.exit(1)

    parsed_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(parsed_file, 'r') as f:
        data = json.load(f)

    plot_data(data, output_file)