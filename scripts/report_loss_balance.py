"""Rebuild fixed loss-control evidence from saved JSON; never load arrays/models."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES=('replay','balanced','force-only')
KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')
COLORS=('#38598b','#1b8766','#e38b22')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text())
    compact={**source,'config':config,'source_run_directory':str(root),'source_summary_sha256':digest(root/'summary.json'),
        'report_generator_sha256':digest(Path(__file__)),'matplotlib_version':matplotlib.__version__}
    arms=source['arms'];compact['attempted_arms']=sum(bool(a.get('executed')) for a in arms.values())
    compact['completed_arms']=sum(a.get('status')=='completed' for a in arms.values())
    compact['blocked_arms']=sum(a.get('status')=='blocked' for a in arms.values())
    compact['artifacts_sha256']={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    compact['metric_convention']='Raw energy MSE over frames; force MSE over atom-coordinate components; fractions use immutable initial errors. Equal-group errors average exhaustive group MSEs.'
    compact['figures_sha256']={};paths=[]
    # Measured observations only: retain all scheduled points, never a selected checkpoint.
    if any(a.get('history') for a in arms.values()):
        figs=output/'figures';figs.mkdir(exist_ok=True)
        for key,label in zip(KEYS,('Energy','Force')):
            fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
            for row,role in enumerate(('fitting','anchor')):
                for col,xkey in enumerate(('update','elapsed_seconds')):
                    ax=axes[row,col]
                    for name,color in zip(NAMES,COLORS):
                        gate='fit_gate' if role=='fitting' else 'anchor_gate'
                        points=[(h[xkey],h.get(gate,{}).get('fraction_remaining',{}).get(key)) for h in arms.get(name,{}).get('history',[]) if xkey in h]
                        points=[(x,y) for x,y in points if y is not None and y>0]
                        if points:ax.scatter(*zip(*points),label=name,color=color,s=16)
                    ax.axhline(.1,color='#666666',ls='--',lw=1,label='gate <0.10');ax.set_yscale('log');ax.grid(alpha=.2)
                    ax.set_xlabel('Optimizer updates' if col==0 else 'Elapsed seconds at measurement')
                    ax.set_ylabel(f'{label} MSE / immutable initial MSE');ax.set_title('128 fitted frames' if row==0 else '16 fitted anchor frames');ax.legend(fontsize=8)
            fig.suptitle(f'Fixed loss controls: {label.lower()} — one seed, measured points only')
            path=figs/f'loss-balance-{label.lower()}.png';fig.savefig(path,dpi=180);plt.close(fig);paths.append(path)
        compact['figures_sha256']={str(p.relative_to(output)):digest(p) for p in paths}
    (output/'loss-balance-diagnostics.json').write_text(json.dumps(compact,sort_keys=True,indent=2,allow_nan=False)+'\n')
    valid=bool(source.get('execution_valid'));balanced=bool(valid and source.get('balanced_joint_accepted'));force=bool(valid and source.get('force_only_force_accepted'))
    lines=['# Controlled fixed loss-balance diagnostics','',f"Matrix: **{source['status']}**; operationally valid: **{valid}**. Full pilot: **pending**.",'',
        f'Balanced joint acceptance: **{balanced}**. Force-only force acceptance: **{force}**.',
        f"Attempted arms: {compact['attempted_arms']}; completed: {compact['completed_arms']}; blocked: {compact['blocked_arms']}.",'',
        '## Fixed protocol','',
        'One seed (0), float64 CPU, 128 fitting frames and original16 anchor; all 16 anchor records are fitted. Frozen zero-head model: 16 features, degree1, two message-passing iterations, eight radial bases, 5Å cutoff. Adam learning rate0.001, 2000updates /256000training frame exposures per arm, microbatch16, wall cap1800s including initialization, compilation, warmup, scoring and fresh-process checkpoint verification. Warmup128frame exposures are discarded and recorded separately. No early stop, retries, resume, weight search or adaptation.','',
        'Replay weights: energy1/force1. Balanced weights: energy0.014584365670390719/force1.9854156343296092; force-only: energy0/force2. Raw MSEs are divided by unchanged train-only squared loss scales before fixed weighting. Calibration uses fresh-C whole-tree gradient norms, energy72.41148277153111/force0.531916604873417, energy/force ratio136.13314964807924. The whole-tree energy norm includes energy-only bias59.332970471878184; this balances initial whole-tree norm contributions only, not shared-parameter contributions or later gradients/Adam updates.','',
        'Original raw gates remain strict: both energy and force MSE <0.1 immutable initial error on fitting AND anchor for balanced acceptance. Force-only acceptance reports only force on both partitions and cannot clear the joint gate. Force-only leaves the absolute energy gauge unconstrained; the fixed normalization offset is retained, without recentering.','',
        '## Fixed endpoints','',
        '| Arm | Status | Updates | Fit E fraction | Fit F fraction | Anchor E fraction | Anchor F fraction | Fit joint | Anchor joint | Wall seconds | Training exposures |',
        '|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    def fmt(value):return '—' if value is None else f'{value:.9g}'
    for name in NAMES:
        a=arms.get(name,{});ratios=[fmt(a.get(gate,{}).get('fraction_remaining',{}).get(key)) for gate in ('fit_gate','anchor_gate') for key in KEYS]
        gates=[str(a.get(gate,{}).get('accepted',False)) if a.get('execution_valid') else 'invalid/blocked' for gate in ('fit_gate','anchor_gate')]
        lines += [f"| {name} | {a.get('status','missing')} | {a.get('updates',0)} | {' | '.join(ratios+gates)} | {fmt(a.get('elapsed_seconds'))} | {a.get('frame_exposures',0)} |"]
    if source.get('replay'):lines+=['',f"Replay comparison: `{json.dumps(source['replay'],sort_keys=True)}`."]
    for name in NAMES:
        a=arms.get(name,{})
        lines+=['',f"### {name}",'',f"Stop: `{a.get('stop_reason','not recorded')}`; failure: `{a.get('failure_reason','none')}`; checkpoint fresh-process verification: {a.get('checkpoint_new_process_verified',False)}.",
            f"Compilation {fmt(a.get('compilation_seconds'))}s; warmup {fmt(a.get('warmup_seconds'))}s; synchronized updates {fmt(a.get('steady_update_seconds_total'))}s; scoring {fmt(a.get('evaluation_seconds_total'))}s; evaluation frames {a.get('evaluation_frame_count',0)}; warmup exposures {a.get('warmup_frame_exposures',0)}."]
        for phase in ('initial_metrics','final_metrics'):
            if a.get(phase):
                lines+=['',f'{phase}:','', '| Partition/group | Records | Energy MSE (eV²) | Force MSE (eV²/Å²) |','|---|---:|---:|---:|']
                for role,metrics in a[phase].items():
                    if not isinstance(metrics,dict):continue
                    lines += [f"| {role} | {metrics.get('record_count','—')} | {fmt(metrics.get(KEYS[0]))} | {fmt(metrics.get(KEYS[1]))} |"]
                    for group,values in metrics.get('per_group',{}).items():
                        lines += [f"| {role}/{group} | {values.get('record_count','—')} | {fmt(values.get(KEYS[0]))} | {fmt(values.get(KEYS[1]))} |"]
                    if 'equal_group' in metrics:lines += [f"Equal-group {role}: `{json.dumps(metrics['equal_group'],sort_keys=True)}`."]
    lines+=['','## Interpretation and limitations','']
    if not valid:lines += ['Scientific interpretation blocked: execution, source identity or replay failed. Resolve recorded operational failures before comparing arms.',f"Failure: {source.get('failure_reason','see arm records')}."]
    elif balanced:lines += ['The balanced arm passed both raw gates on fitting and anchor at the fixed2000update endpoint. This motivates a separately reviewed pilot amendment; it does not authorize the full pilot or establish held-out performance.']
    else:lines += ['The balanced arm did not clear both joint gates at the fixed endpoint. Retain all three prescribed outcomes and review the measured tradeoffs before any new intervention. Force-only improvement is a separate ablation result.']
    lines+=['','No causal attribution to Adam, no capacity theorem, no generalization claim: one frozen coefficient pair, one seed and a fully fitted anchor. Equality of initial whole-tree gradient norms does not imply equality of optimizer steps or sustained balance. These results do not decide other weights, architectures, optimizers or larger budgets. All scheduled observations and failed/blocked trials are retained.','',
        f"Next step: `{source.get('next_step','resolve_execution_failure')}`.",'','## Evidence','',
        f"Run `{root}`; summary SHA256 `{compact['source_summary_sha256']}`; generator SHA256 `{compact['report_generator_sha256']}`.",
        'Complete raw metrics, histories, calibration/source identities, initial hashes, per-group errors and artifact checksums: [loss-balance-diagnostics.json](loss-balance-diagnostics.json). Trial rationale/settings/outcomes/stops: [PAPER_EVIDENCE_LOG.md](PAPER_EVIDENCE_LOG.md), BALANCE entries. Rebuild from JSON only: `.venv/bin/python scripts/report_loss_balance.py`.','']
    for path in paths:lines += [f'![{path.stem}](figures/{path.name})','']
    (output/'loss-balance-diagnostics.md').write_text('\n'.join(lines))


if __name__=='__main__':generate(ROOT/'runs/aspirin-loss-balance-2026-09-30',ROOT/'reports')
