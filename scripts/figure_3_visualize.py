import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

def plot_data(data, output_path):
    workloads = ['Solo', 'Random', 'Congested']
    algorithms = ['Gs-Seq-CPU', 'Mw-Seq-CPU']

    gs_percentages = []
    mw_percentages = []

    # Extracting percentages based on Core Time
    for workload in workloads:
        gs_core = data['Gs-Seq-CPU-Profile'][workload]['Core']
        gs_random = data['Gs-Seq-CPU-Profile'][workload]['Random Access']
        gs_percentages.append((gs_random / gs_core) * 100)

        mw_core = data['Mw-Seq-CPU-Profile'][workload]['Core']
        mw_random = data['Mw-Seq-CPU-Profile'][workload]['Random Access']
        mw_percentages.append((mw_random / mw_core) * 100)

    gs_other = [100 - p for p in gs_percentages]
    mw_other = [100 - p for p in mw_percentages]

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), dpi=300)

    x_positions = [0, 0.5, 1.0]
    bar_width = 0.4
    edgecolor = 'black'
    linewidth = 1.5

    title_fontsize = 14
    label_fontsize = 14
    tick_fontsize = 14
    text_fontsize = 13
    legend_fontsize = 14

    # GS Algorithm Plot
    ax = axs[0]
    ax.bar(x_positions, gs_percentages, color='white',
           edgecolor=edgecolor, linewidth=linewidth, width=bar_width, zorder=2)
    ax.bar(x_positions, gs_other, bottom=gs_percentages, color='dimgray',
           edgecolor=edgecolor, linewidth=linewidth, width=bar_width, zorder=2)

    for i, v in enumerate(gs_percentages):
        ax.text(x_positions[i], v / 2, f"{v:.1f}%", ha='center',
                fontweight='bold', fontsize=text_fontsize, zorder=3)

    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage', fontweight='bold', fontsize=label_fontsize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(workloads, fontweight='bold', fontsize=tick_fontsize)
    ax.set_title('GS Algorithm', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(axis='y', zorder=1)

    # MW Algorithm Plot
    ax = axs[1]
    ax.bar(x_positions, mw_percentages, color='white',
           edgecolor=edgecolor, linewidth=linewidth, width=bar_width, zorder=2)
    ax.bar(x_positions, mw_other, bottom=mw_percentages, color='dimgray',
           edgecolor=edgecolor, linewidth=linewidth, width=bar_width, zorder=2)

    for i, v in enumerate(mw_percentages):
        ax.text(x_positions[i], v / 2, f"{v:.1f}%", ha='center',
                fontweight='bold', fontsize=text_fontsize, zorder=3)

    ax.set_ylim(0, 100)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(workloads, fontweight='bold', fontsize=tick_fontsize)
    ax.set_title('MW Algorithm', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.grid(axis='y', zorder=1)

    # Legend
    leg1 = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=linewidth, label='Random Access')
    leg2 = mpatches.Patch(facecolor='dimgray', edgecolor='black', linewidth=linewidth, label='Other Operations')
    fig.legend(handles=[leg1, leg2], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=legend_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python visualize.py parsed_data.json data/figures/figure_3.png")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(json_file, 'r') as f:
        data = json.load(f)

    plot_data(data, output_file)