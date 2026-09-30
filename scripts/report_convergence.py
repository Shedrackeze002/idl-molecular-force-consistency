"""JSON-only continuation accounting; no model, optimizer or dataset evaluation."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES=('replay','balanced','force-only')
KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')
ROLES=('fitting','anchor')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_window(history,window,end_update=6000,threshold=.01):
    result={r:{} for r in ROLES}
    for r in ROLES:
        for k in KEYS:
            points={row['update']:row.get('metrics',{}).get(r,{}).get(k) for row in history}
            a=points.get(end_update-window);b=points.get(end_update);rel=None
            if isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and a>0:rel=(a-b)/a
            result[r][k]={'start_update':end_update-window,'end_update':end_update,'start_mse':a,'end_mse':b,'relative_reduction':rel,
                         'label':'unavailable' if rel is None else ('small terminal-window change' if abs(rel)<threshold else 'larger terminal-window change')}
    return result


def expected_schedule(c):
    start,end,interval=c['start_update'],c['end_update'],c['validation_interval']
    return sorted({start,start+1,end,*range((start//interval+1)*interval,end+1,interval)})


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def match(a,b):return finite(a) and finite(b) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-8)


def gate(metrics,baseline,fraction):
    ratios={k:metrics[k]/baseline[k] if baseline[k]>0 else None for k in KEYS}
    return {'accepted':all(v is not None and v<fraction for v in ratios.values()),'fraction_remaining':ratios,
            'threshold_mse':{k:baseline[k]*fraction for k in KEYS},'gate_fraction':fraction}


def verified_arm(a,c):
    try:
        start,end=c['start_update'],c['end_update'];extra=end-start;fit=a['fitting_frame_count'];anchor=a['anchor_frame_count']
        if (a.get('status')!='completed' or not a.get('executed') or not a.get('execution_valid') or a.get('stop_reason')!='fixed_budget'
            or not a.get('source_hashes_unchanged') or a.get('source_state_hashes_before')!=a.get('source_state_hashes_after')
            or not a.get('source_state_hashes_before') or not a.get('warmup_source_unchanged') or not a.get('oracle_verified')
            or not a.get('checkpoint_new_process_verified') or a.get('pending_update_unknown') or extra!=c['additional_updates']
            or a.get('updates')!=end or a.get('persisted_update')!=end or a.get('additional_updates')!=extra or a.get('attempted_updates')!=extra
            or a.get('additional_frame_exposures')!=extra*fit or a.get('frame_exposures')!=end*fit or a.get('warmup_frame_exposures')!=fit
            or not finite(a.get('elapsed_seconds')) or a['elapsed_seconds']>=c['continuation_wall_seconds']):return False
        h=a['history'];sched=expected_schedule(c)
        if [r.get('update') for r in h]!=sched or a.get('evaluation_passes')!=len(sched) or a.get('evaluation_frame_count')!=len(sched)*(fit+anchor):return False
        v=a['checkpoint_verification'];norm=c['normalization']
        if (v.get('status')!='passed' or v.get('update')!=end or v.get('adam_count')!=end or v.get('normalization')!=norm
            or v.get('state_hashes')!=a.get('final_state_hashes') or v.get('checkpoint_sha256')!=a.get('checkpoint_sha256')
            or not isinstance(v.get('checkpoint_sha256'),str) or len(v['checkpoint_sha256'])!=64
            or not finite(v.get('maximum_absolute_error')) or not 0<=v['maximum_absolute_error']<=1e-10 or v.get('tolerance')!=1e-10
            or v.get('fitting_record_count')!=fit or v.get('anchor_record_count')!=anchor or a['provenance']['normalization']!=norm):return False
        for row in h:
            u=row['update']
            if row.get('additional_updates')!=u-start or row.get('additional_frame_exposures')!=(u-start)*fit or row.get('frame_exposures')!=u*fit:return False
            for r,label in zip(ROLES,('fit_gate','anchor_gate')):
                m=row['metrics'][r];base=a['original_baselines'][r]
                if any(not finite(m[k]) or m[k]<0 or not finite(base[k]) or base[k]<=0 for k in KEYS):return False
                if row[label]!=gate(m,base,c['gate_fraction']) or row['original_baseline_fraction'][r]!=row[label]['fraction_remaining']:return False
                loss=c['loss_weights']['energy']*m[KEYS[0]]/norm['energy_loss_scale_eV']**2+c['loss_weights']['force']*m[KEYS[1]]/norm['force_loss_scale_eV_per_A']**2
                if not match(row['weighted_loss'][r],loss):return False
        for r in ROLES:
            for k in KEYS:
                if not match(h[1]['metrics'][r][k],a['oracle_metrics'][r][k]):return False
        return a['final_metrics']==h[-1]['metrics'] and a['fit_gate']==h[-1]['fit_gate'] and a['anchor_gate']==h[-1]['anchor_gate']
    except (KeyError,TypeError,ValueError,ZeroDivisionError):return False


def verify_historical_pins(c,sources):
    try:
        for group,directory in [('source_sha256','source_run'),('loss_balance_sha256','loss_balance_run'),('optimizer_diagnostic_sha256','optimizer_diagnostic_run')]:
            for p,wanted in c[group].items():
                if digest(Path(c[directory])/p)!=wanted:return False
        if sum(len(c[g]) for g in ('source_sha256','loss_balance_sha256','optimizer_diagnostic_sha256'))+len(sources['sources']['calibration'])!=118:return False
        for p,wanted in sources['sources']['calibration'].items():
            if digest(Path(c['calibration_run'])/p)!=wanted:return False
        for path,key in [('calibration_path','calibration_sha256'),('loss_balance_compact_path','loss_balance_compact_sha256'),('optimizer_diagnostic_compact_path','optimizer_diagnostic_compact_sha256')]:
            if digest(c[path])!=c[key]:return False
        return True
    except (OSError,KeyError,TypeError,ValueError):return False


def verify_files(root,source,c):
    """Tie all report assertions to saved raw records and historical pins."""
    try:
        sources=json.loads((root/'sources.json').read_text());records=json.loads((root/'optimizer-records.json').read_text())
        if not verify_historical_pins(c,sources):return False
        for name,a in source['arms'].items():
            path=root/name;ac=json.loads((path/'config.json').read_text());prov=json.loads((path/'provenance.json').read_text())
            if (ac!={**c,'normalization':prov['normalization'],'arm':name,'loss_weights':c['arms'][name]}
                or json.loads((path/'summary.json').read_text())!=a or json.loads((path/'history.json').read_text())!=a['history']
                or json.loads((path/'checkpoint-verification.json').read_text())!=a['checkpoint_verification']
                or a['provenance']!=prov or a['original_baselines']!=sources['arms'][name]['initial_metrics']
                or a['source_state_hashes_before']!=records[name]['state_hashes_before'] or a['checkpoint_sha256']!=digest(path/'last.msgpack')):return False
            contract=json.loads((path/'checkpoint-contract.json').read_text());oracle=json.loads((path/'oracle-verification.json').read_text())
            expected=next(p['metrics'] for p in records[name]['probes'] if p['direction']=='stateful_adam' and p['scale']==1.)
            if (contract['checkpoint_sha256']!=a['checkpoint_sha256'] or contract['state_hashes']!=a['final_state_hashes']
                or contract['update']!=c['end_update'] or contract['adam_count']!=c['end_update'] or contract['normalization']!=prov['normalization']
                or oracle['status']!='passed' or oracle['metrics']!=expected or a['oracle_metrics']!=expected):return False
        return True
    except (OSError,KeyError,TypeError,StopIteration,ValueError):return False


def generate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    s=json.loads((root/'summary.json').read_text());c=json.loads((root/'config.json').read_text());arms=s['arms']
    compact={**s,'config':c,'source_run_directory':str(root),'source_summary_sha256':digest(root/'summary.json'),
             'report_generator_sha256':digest(__file__),'full_pilot':'pending','matplotlib_version':matplotlib.__version__}
    arm_configs={n:{**c,'normalization':a.get('provenance',{}).get('normalization',c.get('normalization',{})),
                    'loss_weights':c.get('arms',{}).get(n,c.get('loss_weights',{}))} for n,a in arms.items()}
    compact['interpretation_valid']=bool(s.get('status')=='completed' and s.get('execution_valid') and s.get('source_hashes_unchanged')
        and set(arms)==set(NAMES) and all(verified_arm(a,arm_configs[n]) for n,a in arms.items()) and verify_files(root,s,c))
    compact['learning_accepted']=bool(compact['interpretation_valid'] and any(arms[n]['fit_gate']['accepted'] and arms[n]['anchor_gate']['accepted'] for n in ('replay','balanced')))
    compact['force_only_force_accepted']=bool(compact['interpretation_valid'] and all(arms['force-only']['final_metrics'][r][KEYS[1]]/arms['force-only']['original_baselines'][r][KEYS[1]]<c['gate_fraction'] for r in ROLES))
    compact['actual_additional_updates']=sum(a.get('additional_updates',0) for a in arms.values())
    compact['actual_additional_frame_exposures']=sum(a.get('additional_frame_exposures',0) for a in arms.values())
    compact['arms']={n:{**a,'frame_exposures':a.get('frame_exposures'),
                       'terminal_window':terminal_window(a.get('history',[]),c['terminal_window_updates'],c['end_update'],c['small_window_relative_change'])} for n,a in arms.items()}
    compact['artifacts_sha256']={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    plots=[]
    if any(a.get('history') for a in arms.values()):
        figures=output/'figures';figures.mkdir(exist_ok=True)
        for key,label in zip(KEYS,('energy','force')):
            fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
            for col,name in enumerate(NAMES):
                a=arms.get(name,{})
                for row,r in enumerate(ROLES):
                    ax=axes[row,col];points=[(x['update'],x['metrics'][r][key]) for x in a.get('history',[])]
                    if points:ax.plot(*zip(*points),marker='.',lw=1,label='Prescribed observations')
                    initial=a.get('original_baselines',{}).get(r,{}).get(key)
                    if initial is not None:ax.axhline(initial*c['gate_fraction'],ls='--',color='#b44942',label='Original 10% gate')
                    ax.set_title(f'{name}: '+('128 fitting' if r=='fitting' else '16 fitted anchor'));ax.set_xlabel('Cumulative committed updates');ax.set_ylabel(f'Raw {label} MSE');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(f'Fixed continuation: {label} — every scheduled observation')
            path=figures/f'convergence-budget-{label}.png';fig.savefig(path,dpi=170);plt.close(fig);plots.append(path)
    compact['figures_sha256']={str(p.relative_to(output)):digest(p) for p in plots}
    (output/'convergence-budget-diagnostics.json').write_text(json.dumps(compact,indent=2,sort_keys=True,allow_nan=False)+'\n')
    fmt=lambda v:'—' if v is None else f'{v:.10g}'
    lines=['# Fixed continuation-budget diagnostics','',f"Execution: **{s['status']}**. Interpretation valid: **{compact['interpretation_valid']}**. Full pilot: **pending**.",'',
           f"Additional committed updates: {compact['actual_additional_updates']}; additional fitting-frame exposures: {compact['actual_additional_frame_exposures']}.",
           'Raw energy MSE is averaged over frames; raw force MSE over atom-coordinate components. Baselines are the original fresh raw errors; energy is never recentered. The anchor is fully fitted, providing no held-out evidence.','',
           'Exactly three saved Adam controls, unchanged128fitting/16anchor, seed0/float64CPU/LR.001/16features and original loss weights. Source2000→6000,4000additional updates per arm,128exposures per update.42scheduled measurements, including2001 reconciled to the prior actual Adam probe; warmup proposals discarded. No best checkpoint or learning early stop. Per-arm1200s/controller3900s boundary caps include verification/output.','',
           '| Arm | Status | Global updates | Additional updates | New exposures | Cumulative exposures | Joint original gate |','|---|---|---:|---:|---:|---:|---|']
    for n in NAMES:
        a=arms.get(n,{})
        lines.append(f"| {n} | {a.get('status','missing')} | {a.get('updates','—')} | {a.get('additional_updates',0)} | {a.get('additional_frame_exposures',0)} | {a.get('frame_exposures','—')} | {bool(a.get('fit_gate',{}).get('accepted') and a.get('anchor_gate',{}).get('accepted'))} |")
    lines+=['','| Arm | Role | Raw E MSE | Raw F MSE | Original E fraction | Original F fraction | E window reduction | F window reduction |','|---|---|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        a=compact['arms'].get(n,{})
        for r in ROLES:
            m=a.get('final_metrics',{}).get(r,{});base=a.get('original_baselines',{}).get(r,{})
            values=[m.get(k) for k in KEYS]+[m[k]/base[k] if m.get(k) is not None and base.get(k,0)>0 else None for k in KEYS]+[a.get('terminal_window',{}).get(r,{}).get(k,{}).get('relative_reduction') for k in KEYS]
            lines.append(f'| {n} | {r} | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['',f"Replay/balanced joint learning accepted: **{compact['learning_accepted']}**. Force-only force gate: **{compact['force_only_force_accepted']}**, separate from joint acceptance/full pilot.",'',
            'Terminal reduction is signed(MSE5500−MSE6000)/MSE5500. Absolute change<1% is labelled only “small terminal-window change”; absent/zero baseline is unavailable. These labels do not establish convergence, stationarity, capacity limits, optimizer causality or eventual success. Budget sensitivity in this one fitted setting remains distinct from generalization and physical landscape recovery.','',
            'Missing/failed execution, counter/history/oracle/source/receipt or baseline verification blocks interpretation. Partial synchronized counts remain actual counts; pending interrupted work is unknown. Failed snapshots and all prescribed points are retained. No numerical retry or later intervention is automatic.','']
    for n,a in arms.items():lines.append(f"- {n}: failure/stop {a.get('failure_reason',a.get('stop_reason','unavailable'))}; synchronized update {a.get('updates','unavailable')}, persisted update {a.get('persisted_update','unavailable')}, pending update unknown {a.get('pending_update_unknown','unavailable')}.")
    lines+=['',f"Source summary SHA256 `{compact['source_summary_sha256']}`; report generator SHA256 `{compact['report_generator_sha256']}`.",
            'Complete raw/per-group/equal-group/overlap metrics, original fractions, weighted losses, actual histories, state/probe/source verification and checksums: [convergence-budget-diagnostics.json](convergence-budget-diagnostics.json). Trial trail: [PAPER_EVIDENCE_LOG.md](PAPER_EVIDENCE_LOG.md).','']
    for p in plots:lines.extend([f'![{p.stem}](figures/{p.name})',''])
    (output/'convergence-budget-diagnostics.md').write_text('\n'.join(lines))
    return compact


if __name__=='__main__':generate(ROOT/'runs/aspirin-convergence-budget-2026-09-30',ROOT/'reports')
