"""Rebuild the coverage report/figures from saved diagnostics only (no data loader)."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/aspirin-data-coverage-2026-09-30'
OUT=ROOT/'reports'
KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')
NAMES=('narrow16','diverse16','diverse128')
COLORS=('#38598b','#e38b22','#1b8766')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=json.loads((RUN/'summary.json').read_text())
    compact={k:v for k,v in source.items() if k not in ('arms','provenance')}
    prov=source.get('provenance',{})
    compact['provenance']={k:v for k,v in prov.items() if k not in ('code_files','historical_reference','training_arrays')}
    compact['reference_hashes']=prov.get('historical_reference',{}).get('hashes')
    compact['source_run_directory']=str(RUN.relative_to(ROOT));compact['source_summary_sha256']=digest(RUN/'summary.json')
    compact['arms']={}
    for name,arm in source['arms'].items():
        compact['arms'][name]={k:v for k,v in arm.items() if k!='provenance'}
        compact['arms'][name]['artifacts_sha256']={p.name:digest(p) for p in sorted((RUN/name).iterdir()) if p.is_file()}
    target=OUT/'data-coverage-diagnostics.json'
    target.write_text(json.dumps(compact,sort_keys=True,indent=2,allow_nan=False)+'\n')
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    paths=[]
    for key,label in zip(KEYS,('Energy','Force')):
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for row,metric_scope in enumerate(('fitting','anchor')):
            for col,xkey in enumerate(('update','elapsed_seconds')):
                ax=axes[row,col]
                for name,color in zip(NAMES,COLORS):
                    history=compact['arms'][name].get('history',[])
                    points=[(r[xkey],r[f'{"fit" if metric_scope=="fitting" else "anchor"}_gate']['fraction_remaining'][key]) for r in history]
                    points=[(x,y) for x,y in points if y is not None and y>0]
                    if points: ax.scatter(*zip(*points),color=color,s=16,label=name)
                ax.axhline(.1,ls='--',color='#666666',lw=1,label='gate: <0.10' if row==0 and col==0 else None)
                ax.set_yscale('log');ax.set_xlabel('Optimizer updates' if col==0 else 'Elapsed seconds at measurement')
                ax.set_ylabel(f'{label} MSE / immutable initial MSE')
                ax.set_title('Fitting subset' if row==0 else 'Original16 anchor');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'{label} fitting and common-anchor diagnostics (single seed; measured points only)')
        path=figs/f'data-coverage-{label.lower()}.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    history=compact['arms']['diverse16'].get('history',[])
    for ax,xkey in zip(axes,('update','elapsed_seconds')):
        for scope,color,label in [('anchor_fitted','#38598b','2 fitted anchor frames'),('anchor_not_fitted','#e38b22','14 anchor frames not fitted by B')]:
            points=[(r[xkey],r['metrics'][scope][KEYS[1]]) for r in history if r['metrics'][scope][KEYS[1]] is not None]
            if points: ax.scatter(*zip(*points),color=color,s=18,label=label)
        ax.set_xlabel('Optimizer updates' if xkey=='update' else 'Elapsed seconds at measurement');ax.set_ylabel('Force MSE (eV²/Å²)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Diverse16 anchor overlap — all records belong to the frozen training partition')
    path=figs/'data-coverage-anchor-overlap.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
    compact['figures_sha256']={str(p.relative_to(ROOT)):digest(p) for p in paths}
    target.write_text(json.dumps(compact,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# Training-data coverage diagnostics','',f"Matrix execution: **{compact['status']}**. Full pilot remains **pending**.",'',
           'These are training-only engineering controls, one seed per condition. They do not establish generalization or sample-count causality. Previous failed learning checks remain unchanged.','',
           '## Fixed protocol','',
           'Seed0, float64 CPU, zero energy head,16features/degree1/two message-passing iterations/eight radial basis functions/5Å cutoff, Adam1e-3 and unchanged train-only normalization. Each arm has2000updates/1800s including initialization, compilation, warmup, scoring and reload verification, with no learning early stop. C accumulates eight16-frame chunks at unchanged incoming parameters before one optimizer update.','',
           'A fits the original16 from connection01. B fits one frame per branch on eight connections (16total) and overlaps only2anchor records. C fits16frames per connection on the same eight groups (128total), containing both16sets. Support:01/23/77/58_2/44/22/78_2/01_2. Selection is label-blind and frozen.','',
           '## Outcomes','',
           '| Arm | Updates | Fit E fraction | Fit F fraction | Anchor E fraction | Anchor F fraction | Fit gate | Anchor gate | Wall seconds | Training frame exposures |',
           '|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    def ratio(arm,gate,key):
        v=arm.get(gate,{}).get('fraction_remaining',{}).get(key)
        return f'{v:.8g}' if v is not None else '—'
    for name in NAMES:
        a=compact['arms'][name]
        lines.append(f"| {name} | {a['updates']} | {ratio(a,'fit_gate',KEYS[0])} | {ratio(a,'fit_gate',KEYS[1])} | {ratio(a,'anchor_gate',KEYS[0])} | {ratio(a,'anchor_gate',KEYS[1])} | {'pass' if a['fit_gate']['accepted'] else 'fail/not valid'} | {'pass' if a['anchor_gate']['accepted'] else 'fail/not valid'} | {a.get('elapsed_seconds',0):.3f} | {a.get('frame_exposures',0)} |")
    lines+=['', 'Both energy and force MSE must independently remain below0.10of their immutable initial values to pass a gate. Fit gates use their own subset initial values. All anchor gates use fixed original16initial energy0.061298358531283896eV² and force0.010133561081129106eV²/Å². Empty partitions/zero initial errors yield null ratios and cannot pass.','']
    if compact.get('replay'):
        lines += [f"A replay: {compact['replay']['status']}; maximum absolute MSE difference {compact['replay']['maximum_absolute_mse_difference']:.17g}, using atol1e-10/rtol1e-8 at every historical100-update measurement through2000.",'']
    for name in NAMES:
        a=compact['arms'][name]
        lines += [f"**{name}:** {a['status']}; stop `{a.get('stop_reason')}`; all-frame fresh-process checkpoint verification: {a.get('checkpoint_new_process_verified',False)}. First observed fitting gate passage: {a.get('first_observed_fit_passing_scheduled_update','none')}; anchor gate passage: {a.get('first_observed_anchor_passing_scheduled_update','none')}.",'']
        if a.get('final_metrics'):
            fit=a['final_metrics']['fitting'];anchor=a['final_metrics']['anchor']
            lines += [f"Final fitting E MSE={fit[KEYS[0]]:.17g}, F MSE={fit[KEYS[1]]:.17g}; original16E MSE={anchor[KEYS[0]]:.17g}, F MSE={anchor[KEYS[1]]:.17g}. Compilation={a.get('compilation_seconds',0):.3f}s; warmup={a.get('warmup_seconds',0):.3f}s; synchronized updates={a.get('steady_update_seconds_total',0):.3f}s; scoring={a.get('evaluation_seconds_total',0):.3f}s ({a.get('evaluation_frame_count',0)}frame evaluations). Warmup exposures are separately recorded and excluded from training exposure counts.",'']
        if a.get('failure_reason'): lines += [f"Failure reason: {a['failure_reason']}",'']
    lines += ['## Interpretation and next step','']
    if compact['next_step']=='architecture_gradient_investigation':
        lines += ['C did not pass both gates at the fixed budget. Extra coverage did not resolve the original fitting gate in this comparison. Preserve these results and return to a separately planned architecture/loss-gradient investigation. Partial improvements cannot justify selecting a successful run or relaxing the gate.','']
    elif compact['next_step']=='separately_review_protocol_amendment':
        lines += ['C passed both gates at the complete fixed budget. This supports considering a separately saved, twice-reviewed baseline-protocol amendment; it does not automatically clear the old batch4gate or authorize the full pilot.','']
    else: lines += ['Execution/replay did not complete validly; downstream comparisons remain blocked or incomplete. Resolve the recorded operational failure before drawing a coverage conclusion.','']
    lines += ['B changes fitted geometries and difficulty. C adds within-connection coverage and8times more frame exposures per update; equal updates are not equal compute. The elapsed-time plots show actual measurement times and do not estimate unmeasured metrics. The2versus14anchor partition is diagnostic overlap, not a scientific held-out split. No extra seed, GPU/full-pilot/scientific arm, loss/capacity/learning-rate change, longer budget or test evaluation occurred.','',
              '## Evidence and reproduction','',f"- Raw run: `{compact['source_run_directory']}`; summary SHA256 `{compact['source_summary_sha256']}`.",
              f"- Execution revision `{prov.get('revision')}`; source fingerprint `{prov.get('code_fingerprint')}`.",
              f"- Frozen subsets `{prov.get('frozen_subsets_identity')}`; dataset `{prov.get('dataset_identity')}`; split `{prov.get('split_hash')}`.",
              '- Full-precision metrics, complete scheduled histories, initial state hashes and per-file checksums: [data-coverage-diagnostics.json](data-coverage-diagnostics.json).',
              '- Trial preregistrations and all outcomes: [PAPER_EVIDENCE_LOG.md](PAPER_EVIDENCE_LOG.md), COVERAGE records.',
              '- Rebuild this report/figures from saved runs: `.venv/bin/python scripts/report_data_coverage.py`.','']
    for p in paths: lines += [f'![{p.stem}](figures/{p.name})','']
    (OUT/'data-coverage-diagnostics.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
