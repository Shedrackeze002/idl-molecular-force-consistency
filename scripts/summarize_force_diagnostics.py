"""Build paper-facing diagnostic summary/figure from immutable saved runs."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/aspirin-force-diagnostics-2026-09-30'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    result=json.loads((RUN/'summary.json').read_text())
    control={k:{name:v for name,v in result[k].items() if name!='provenance'} for k in ('teacher','real')}
    identities={str(p.relative_to(ROOT)):digest(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    compact={'scope':'Training-only engineering diagnosis; original learning gate failed, full pilot pending',
             'config':json.loads((RUN/'config.json').read_text()),'execution_valid':result['execution_valid'],
             'readout':result['readout'],'controls':control,'provenance':json.loads((RUN/'provenance.json').read_text()),
             'artifact_sha256':identities,'figure_generator':{'sha256':digest(Path(__file__)),'matplotlib':matplotlib.__version__}}
    output=ROOT/'reports/force-learning-diagnostics.json'
    output.write_text(json.dumps(compact,indent=2,sort_keys=True,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    baseline=control['real']['baselines']['zero_force_mse_eV2_per_A2']
    sources=[('Random head, batch 4',ROOT/'runs/aspirin-learning-check-seed0/learning-history.json'),
             ('Zero head, batch 4',ROOT/'runs/aspirin-learning-check-zero-readout-seed0/learning-history.json'),
             ('Zero head, batch 16',RUN/'real/history.json')]
    figure_inputs={}
    for label,path in sources:
        if path.exists():
            h=json.loads(path.read_text())
            axes[0].plot([v['update'] for v in h],[v['force_mse_eV2_per_A2']/baseline for v in h],label=label,lw=1.7)
        else:
            path=path.parent/'summary.json';saved=json.loads(path.read_text())
            h=[{'update':0,**saved['learning_initial_mse']},{'update':saved['updates'],**saved['learning_final_mse']}]
            axes[0].scatter([v['update'] for v in h],[v['force_mse_eV2_per_A2']/baseline for v in h],marker='^',label=label+' (endpoints)')
        figure_inputs[str(path.relative_to(ROOT))]=digest(path)
        if label.startswith('Random'): random_initial=h[0]['force_mse_eV2_per_A2']
    compact['figure_input_sha256']=figure_inputs
    output.write_text(json.dumps(compact,indent=2,sort_keys=True,allow_nan=False)+'\n')
    axes[0].axhline(1.,color='#999999',ls=':',label='Zero-force baseline')
    axes[0].axhline(.1*random_initial/baseline,color='#888888',ls='--',label='Random-head gate')
    axes[0].axhline(.1,color='black',ls='--',label='Zero-head gates')
    axes[0].set(title='Original molecular labels (same 16 frames)',xlabel='Optimizer updates',ylabel='Force MSE / zero-force MSE',yscale='log')
    axes[0].legend(fontsize=8,ncol=2,loc='upper right')
    for key,label in [('energy_mse_eV2','Energy'),('force_mse_eV2_per_A2','Forces')]:
        h=control['teacher']['history'];initial=control['teacher']['initial_mse'][key]
        axes[1].plot([v['update'] for v in h],[v[key]/initial for v in h],marker='o',label=label,lw=1.7)
    axes[1].axhline(.1,color='black',ls='--',label='Acceptance threshold')
    axes[1].set(title='Synthetic teacher (different, in-class labels)',xlabel='Optimizer updates',ylabel='MSE / own initial MSE',yscale='log')
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('Force-learning diagnosis on a fixed training subset',fontsize=13)
    for suffix in ('png','pdf'): fig.savefig(ROOT/f'reports/figures/force-learning-diagnostics.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps({'summary_sha256':digest(output),'teacher_force_reduction_percent':100*(1-control['teacher']['fraction_remaining']['force_mse_eV2_per_A2']),
                      'real_force_reduction_percent':100*(1-control['real']['fraction_remaining']['force_mse_eV2_per_A2']),
                      'real_force_rmse_eV_per_A':control['real']['final_mse']['force_mse_eV2_per_A2']**.5},indent=2))

if __name__=='__main__': main()
