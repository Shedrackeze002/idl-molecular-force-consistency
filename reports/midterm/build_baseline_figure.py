"""Render saved report data only; no model loading or dataset access."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT.parent/'archive/midterm/midsemester-2026-10-18/figures/plot-data.json').read_text())
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

fig, ax = plt.subplots(figsize=(6.2, 1.85), constrained_layout=True)
colors = {'A': '#64748b', 'B': '#c66b25', 'C': '#167477'}
for name in ('A', 'B', 'C'):
    curve = DATA['baseline_curves'][name]
    rows = curve['rows']
    ax.plot([r['update']/1000 for r in rows],
            [1000*r['equal_connection_force_mae_eV_per_A'] for r in rows],
            color=colors[name], label=name, linewidth=1.2)
    ax.plot(curve['best_update']/1000, 1000*curve['best_force_mae_eV_per_A'],
            {'A':'o','B':'s','C':'^'}[name], color=colors[name], markersize=6, markeredgecolor='white', markeredgewidth=.6)
ax.axhline(50, linestyle='--', color='#475569', linewidth=.8, label='50 meV/Å criterion')
ax.set(xlabel='Training updates (thousands)',
       ylabel='Force MAE (meV/Å)', xlim=(0, 200), ylim=(30, 84))
ax.legend(ncol=4, frameon=False, loc='upper right', fontsize=8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT/f'figures/baseline-curves.{suffix}', dpi=180)
plt.close(fig)

print('Rendered baseline curves from saved metric data.')
