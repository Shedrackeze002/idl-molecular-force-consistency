"""Supplementary logarithmic figure; uses saved observations only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.loads(Path('reports/late-rate-diagnostics.json').read_text())
assert r['execution_valid']
fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3))
for name, color in [('constant', '#2456a6'), ('late_decay', '#d36d22')]:
    h = r['arms'][name]['history']
    for ax, key in zip(axs, ('energy_mse_eV2', 'force_mse_eV2_per_A2')):
        ax.plot([v['update'] for v in h],
                [v['fitting_gate']['fraction_remaining'][key] for v in h],
                color=color, marker='o', markersize=2.8, linewidth=1.5,
                label=name.replace('_', ' '))
for ax, title in zip(axs, ('Energy fitting error (log scale)', 'Force fitting error')):
    ax.set_title(title)
    ax.set_xlabel('Absolute optimizer update')
    ax.set_ylabel('MSE / original update-0 MSE')
    ax.axhline(.1, color='#656565', linestyle='--', linewidth=1,
               label='Acceptance threshold')
    ax.axvline(8000, color='#999999', linestyle=':', linewidth=1)
    ax.set_xlim(5950, 10050)
    ax.grid(alpha=.18, which='both')
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_yscale('log')
axs[0].set_ylim(.08, 60)
axs[0].annotate('39.8× baseline at 8,001', xy=(8001, 39.76470431227181),
                xytext=(8400, 28), fontsize=8,
                arrowprops={'arrowstyle': '->', 'color': '#656565'})
axs[1].set_ylim(0, .82)
axs[1].legend(loc='lower left', fontsize=8)
fig.suptitle('Late learning-rate diagnostic: equal 4,000-update continuations', fontsize=13)
fig.text(.5, .015, '128 fitted aspirin frames; one seed; sampled observations. Lines connect scores and do not resolve intervening behavior.', ha='center', fontsize=8)
fig.tight_layout(rect=[0, .045, 1, .94])
out = Path('reports/figures/late-rate-curves-log.png')
fig.savefig(out, dpi=180)
plt.close(fig)
print(out.resolve())
