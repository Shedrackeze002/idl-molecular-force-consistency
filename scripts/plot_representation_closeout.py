"""Plot verified saved-data representation summaries; no numerical model code."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'E64': '#0072B2', 'SP': '#D55E00', 'S64': '#009E73'}
FAMILIES = ('E64', 'SP', 'S64')
METRICS = ('best', 'last', 'late20')
LABELS = {'best': 'BEST checkpoint', 'last': 'LAST checkpoint', 'late20': 'Late20 trajectory mean'}


def save(figure, directory, name):
    figure.savefig(directory/(name+'.png'), dpi=220, bbox_inches='tight')
    figure.savefig(directory/(name+'.pdf'), bbox_inches='tight')
    plt.close(figure)


def plot_all(result, output):
    if result['status'] != 'VERIFIED_SAVED_VALIDATION_ANALYSIS':
        raise ValueError('Only verified saved-data summaries can be plotted')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for index, family in enumerate(FAMILIES):
            values = np.asarray(result['families'][family][metric]['seed_values'])*1000
            for seed, marker in [(0, 'o'), (1, 's')]:
                axis.scatter(index + (-.075 if seed == 0 else .075), values[seed],
                             marker=marker, color=COLORS[family], s=48,
                             label=f'seed {seed}' if index == 0 else None, zorder=3)
            axis.plot([index-.19, index+.19], [values.mean()]*2, color='black', lw=1.6)
        axis.set_xticks(range(3), ['E64', 'SP', 'S64'])
        axis.set_title(LABELS[metric]);axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Equal-connection validation force MAE (meV/Å)')
    axes[0].legend(frameon=False)
    figure.suptitle('Fixed E64-selected recipe; two seeds per family', y=1.01, fontsize=12)
    figure.text(.5, -.02, 'Dots/squares show individual seeds; black marks show arithmetic means, not confidence intervals.',
                ha='center', fontsize=9)
    figure.text(.5, -.07, 'Parameter counts: E64 43,521; SP 43,473; S64 25,473.', ha='center', fontsize=9)
    figure.tight_layout()
    save(figure, output, 'validation_summaries')

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for axis, family in zip(axes, ('SP', 'S64')):
        for index, metric in enumerate(METRICS):
            values = np.asarray(result['pairs'][family][metric]['seed_deltas'])*1000
            axis.plot([index, index], [values.min(), values.max()], color=COLORS[family], alpha=.55)
            for seed, marker in [(0, 'o'), (1, 's')]:
                axis.scatter(index+(-.055 if seed == 0 else .055), values[seed], marker=marker,
                             color=COLORS[family], s=48, label=f'seed {seed}' if index == 0 else None, zorder=3)
        axis.axhline(0, color='black', lw=.8)
        axis.set_xticks(range(3), ['BEST', 'LAST', 'Late20'])
        axis.set_title(('Primary: ' if family == 'SP' else 'Secondary: ')+family+' minus E64')
        axis.grid(axis='y', alpha=.2);axis.legend(frameon=False)
    axes[0].set_ylabel('Paired force MAE difference (meV/Å)')
    figure.text(.5, -.02, 'Positive values mean lower E64 error. Lines join the two observed seeds, not confidence bounds.',
                ha='center', fontsize=9)
    figure.tight_layout()
    save(figure, output, 'paired_seed_differences')

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in FAMILIES:
        for seed in (0, 1):
            name = f'{family}-{seed}'
            rows = [row for row in result['trajectories'] if row['candidate'] == name]
            x, y = [r['update'] for r in rows], [r['force_mae_meV_per_A'] for r in rows]
            axes[0].plot(x, y, color=COLORS[family], ls='-' if seed == 0 else '--', lw=.8,
                         alpha=.8, label=name)
            late = [r for r in rows if r['update'] >= 150000]
            axes[1].plot([r['update'] for r in late], [r['force_mae_meV_per_A'] for r in late],
                         color=COLORS[family], ls='-' if seed == 0 else '--', lw=.85, alpha=.8)
    for axis in axes:
        axis.set_xlabel('Training updates');axis.grid(alpha=.15)
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    axes[0].set_ylabel('Validation force MAE (meV/Å)')
    axes[0].set_title('All 401 scheduled observations; no smoothing')
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    axes[1].set_title('Final 50,000 updates')
    axes[1].axvspan(190500, 200000, color='grey', alpha=.10)
    figure.text(.5, -.02, 'Shaded region is the declared late20 window. Each panel includes its full observed error range.',
                ha='center', fontsize=9)
    figure.tight_layout()
    save(figure, output, 'validation_histories')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_all(json.loads(args.analysis.read_text()), args.output)
