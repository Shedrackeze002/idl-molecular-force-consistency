"""Render saved report data only; no model loading or dataset access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'figures/plot-data.json').read_text())
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

fig, ax = plt.subplots(figsize=(6.2, 2.05), constrained_layout=True)
colors = {'A': '#64748b', 'B': '#c66b25', 'C': '#167477'}
for name in ('A', 'B', 'C'):
    curve = DATA['baseline_curves'][name]
    rows = curve['rows']
    ax.plot([r['update']/1000 for r in rows],
            [1000*r['equal_connection_force_mae_eV_per_A'] for r in rows],
            color=colors[name], label=name, linewidth=1.2)
    ax.plot(curve['best_update']/1000, 1000*curve['best_force_mae_eV_per_A'],
            'o', color=colors[name], markersize=4)
ax.axhline(50, linestyle='--', color='#475569', linewidth=.8, label='A1 threshold')
ax.set(xlabel='Training updates (thousands)',
       ylabel='Force MAE (meV/Å)', xlim=(0, 200), ylim=(30, 84))
ax.legend(ncol=4, frameon=False, loc='upper right', fontsize=8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT/f'figures/baseline-curves.{suffix}', dpi=180)
plt.close(fig)

names = ['energy', 'direct', 'consistency-0.1', 'consistency-1', 'consistency-10']
labels = ['Energy', 'Direct', 'λ = 0.1', 'λ = 1', 'λ = 10']
fig, ax = plt.subplots(figsize=(6.2, 1.95), constrained_layout=True)
for index, name in enumerate(names):
    row = DATA['controlled_summaries'][name]
    values = 1000*np.array(row['seed_force_mae_eV_per_A'])
    ax.plot([index, index], [min(values), max(values)], color='#94a3b8', linewidth=1)
    ax.scatter(index-.07, values[0], marker='o', s=25, color='#167477',
               label='Seed 0' if index == 0 else None)
    ax.scatter(index+.07, values[1], marker='^', s=30, color='#c66b25',
               label='Seed 1' if index == 0 else None)
    ax.plot(index, 1000*row['mean_force_mae_eV_per_A'], '_', color='#0f172a',
            markersize=12, markeredgewidth=1.8)
ax.set(xticks=range(5), xticklabels=labels,
       ylabel='Force MAE (meV/Å)', ylim=(32.5, 41.5))
ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT/f'figures/controlled-force.{suffix}', dpi=180)
plt.close(fig)
print('Rendered two figures from saved, hash-bound metric data.')
