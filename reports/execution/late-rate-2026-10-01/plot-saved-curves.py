"""Static figure from verified saved observations; zero model evaluations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
r=json.loads(Path('reports/late-rate-diagnostics.json').read_text())
if not r['execution_valid']:raise RuntimeError('Complete curves require verified execution')
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3))
colors={'constant':'#2456a6','late_decay':'#d36d22'}
for name,a in r['arms'].items():
 h=a['history'];x=[v['update'] for v in h]
 for ax,key in zip(axs,('energy_mse_eV2','force_mse_eV2_per_A2')):
  y=[v['fitting_gate']['fraction_remaining'][key] for v in h]
  ax.plot(x,y,color=colors[name],marker='o',markersize=2.8,linewidth=1.5,label=name.replace('_',' '))
for ax,title in zip(axs,('Energy fitting error','Force fitting error')):
 ax.set_title(title);ax.set_xlabel('Absolute optimizer update');ax.set_ylabel('Original MSE remaining')
 ax.yaxis.set_major_formatter(PercentFormatter(xmax=1));ax.axhline(.1,color='#656565',linestyle='--',linewidth=1,label='Acceptance threshold')
 ax.axvline(8000,color='#999999',linestyle=':',linewidth=1)
 ax.set_xlim(5950,10050);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
maxima=[max(v['fitting_gate']['fraction_remaining'][key] for a in r['arms'].values() for v in a['history']) for key in ('energy_mse_eV2','force_mse_eV2_per_A2')]
axs[0].set_ylim(0,max(.25,1.1*maxima[0]));axs[1].set_ylim(0,max(.45,1.1*maxima[1]));axs[1].legend(loc='lower left',fontsize=8)
fig.suptitle('Late learning-rate diagnostic: equal 4,000-update continuations',fontsize=13)
fig.text(.5,.015,'128 fitted aspirin frames; one seed; sampled checkpoints. Both runs start at the same update-6,000 state.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.045,1,.94]);out=Path('reports/figures/late-rate-curves.png');out.parent.mkdir(exist_ok=True)
fig.savefig(out,dpi=180);plt.close(fig);print(str(out.resolve()))
