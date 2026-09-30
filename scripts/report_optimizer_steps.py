"""Rebuild local optimizer observations from saved JSON only, including failed work."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES=('replay','balanced','force-only')
DIRECTIONS=('stateful_adam','reset_adam_matched_norm','steepest_matched_norm')
KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cells(case):
    return [cell for d in case.get('directions',{}).values() for r in d.get('roles',{}).values()
            for cell in r.get('finite_differences',{}).get('objectives',{}).values()]


def counts(case):
    cs=cells(case)
    return {'fd_cells':len(cs),'fd_steps':sum(len(c.get('steps',[])) for c in cs),'probes':len(case.get('probes',[]))}


def verified(case):
    if case.get('status')!='completed' or not case.get('verification_valid') or case.get('state_hashes_before')!=case.get('state_hashes_after'):return False
    if not all(case.get(k,{}).get('valid') for k in ('adam_identity','weighted_gradient_identity','builder_identity')):return False
    if case.get('counts')!=counts(case) or case.get('training_frame_exposures')!=0:return False
    for d in case.get('directions',{}).values():
        for r in d.get('roles',{}).values():
            fd=r.get('finite_differences',{})
            if fd.get('status')=='not_applicable_zero_direction':continue
            if not fd.get('valid') or not all(c.get('valid') and any(s.get('valid') for s in c.get('steps',[])) for c in fd.get('objectives',{}).values()):return False
    return True


def generate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text());cases=source['cases']
    compact={**source,'config':config,'source_run_directory':str(root),'source_summary_sha256':digest(root/'summary.json'),
        'report_generator_sha256':digest(__file__),'matplotlib_version':matplotlib.__version__,
        'learning_accepted':False,'full_pilot':'pending','metric_convention':'Raw energy MSE over frames; force MSE over atom-coordinate components; immutable original baseline fractions; fully fitted anchor. Local normalized loss derivatives and counterfactual displacements.'}
    compact['interpretation_valid']=bool(source.get('execution_valid') and source.get('source_hashes_unchanged') and set(cases)==set(NAMES) and all(verified(c) for c in cases.values()))
    compact['attempted_cases']=sum(bool(c.get('executed')) for c in cases.values())
    compact['completed_cases']=sum(c.get('status')=='completed' for c in cases.values());compact['blocked_cases']=sum(c.get('status')=='blocked' for c in cases.values())
    compact['actual_counts']={k:sum(counts(c)[k] for c in cases.values()) for k in ('fd_cells','fd_steps','probes')}
    records=[c.get('compute',{}) for c in cases.values()]
    if (root/'sources.json').exists():
        compact['source_verification']=json.loads((root/'sources.json').read_text());records.append(compact['source_verification']['compute'])
    elif (root/'preflight-partial.json').exists():
        compact['preflight_partial']=json.loads((root/'preflight-partial.json').read_text());records.append(compact['preflight_partial'].get('compute',{}))
    compact['completed_frame_evaluations']={}
    for record in records:
        for category,number in record.get('completed_frame_evaluations',{}).items():
            compact['completed_frame_evaluations'][category]=compact['completed_frame_evaluations'].get(category,0)+number
    compact['unconfirmed_operations']=sum(op.get('status')!='completed' for r in records for op in r.get('operations',[]))
    compact['artifacts_sha256']={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    paths=[]
    if any(c.get('probes') for c in cases.values()):
        figures=output/'figures';figures.mkdir(exist_ok=True)
        for key,label in zip(KEYS,('Energy','Force')):
            fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
            for col,name in enumerate(NAMES):
                for row,role in enumerate(('fitting','anchor')):
                    ax=axes[row,col];case=cases.get(name,{})
                    for direction,color in zip(DIRECTIONS,('#38598b','#1b8766','#e38b22')):
                        points=[(p['scale'],100*p['roles'][role]['source_relative_change'][key]) for p in case.get('probes',[]) if p['direction']==direction and p['roles'][role]['source_relative_change'][key] is not None]
                        if points:ax.scatter(*zip(*points),color=color,label=direction,s=35)
                    ax.axhline(0,color='#666',ls='--',lw=1);ax.set_xticks([.1,1]);ax.set_xlabel('Prescribed displacement scale');ax.grid(alpha=.2)
                    ax.set_ylabel(f'{label} MSE change from source (%)');ax.set_title(f'{name}: '+('128 fitting' if role=='fitting' else '16 fitted anchor'));ax.legend(fontsize=6)
            fig.suptitle(f'Frozen optimizer probes: {label.lower()} — every prescribed measured point')
            p=figures/f'optimizer-step-{label.lower()}.png';fig.savefig(p,dpi=180);plt.close(fig);paths.append(p)
    compact['figures_sha256']={str(p.relative_to(output)):digest(p) for p in paths}
    (output/'optimizer-step-diagnostics.json').write_text(json.dumps(compact,sort_keys=True,indent=2,allow_nan=False)+'\n')
    fmt=lambda v:'—' if v is None else f'{v:.10g}'
    lines=['# Frozen optimizer-step diagnostics','',f"Execution: **{source['status']}**. Interpretation valid: **{compact['interpretation_valid']}**. Full pilot: **pending**.",'',
        f"Attempted cases: {compact['attempted_cases']}; completed: {compact['completed_cases']}; blocked: {compact['blocked_cases']}.",
        f"Recorded FD cells: {compact['actual_counts']['fd_cells']}; FD step records: {compact['actual_counts']['fd_steps']}; counterfactual probes: {compact['actual_counts']['probes']}.",
        f"Committed training frame exposures: 0. Started but unconfirmed operations: {compact['unconfirmed_operations']} (partial work is unknown).",'',
        '## Fixed protocol','',
        'Three saved 2000-update Adam checkpoints, seed0, float64CPU, unchanged128fitting frames and16 fully fitted anchor frames. Same16features/degree1/twoiterations/eightbases/5Å zero-head model, train-only normalization and source weights. Frozen total diagnostic boundary cap600s includes source checks, compilation and output; operations cannot be forcibly cancelled.',
        'The actual saved-moment Adam proposal at learning rate0.001 is verified independently, using incomingcount+1 (ephemeral2001), b1=.9,b2=.999,eps=1e-8,eps_root=0,nesterovFalse. Advanced states are discarded. Fresh-moment Adam and negative weighted-gradient references are rescaled to the actual whole-tree proposal norm. Directions are formed using fitting data, then evaluated unchanged on both roles. Equal whole-tree norms do not equalize perparameter displacement or compute.',
        'Central differences along each available unit direction use bothh1e-4/1e-5 for both normalized objectives on both roles, atol1e-8/rtol1e-4; at least one must pass percell. Both steps are preserved, including failures. Passing tolerance does not certify the sign of a slope close to numerical tolerance. Exactlyscale0.1/1 temporarydisplacements; no winner selection, offset recentering, learning-rate change or persistent update.','',
        'Original raw learning gates remain strict <0.1 of immutable initial MSE. A temporary counterfactual gate cannot clear learning acceptance. All anchor records are fitted; this provides no held-out/generalization evidence.','',
        '## Local directions and finite changes','',
        '| Case | Direction | Role | Direction norm | Energy slope | Force slope | Weighted slope | Energy cosine | Force cosine |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        case=cases.get(name,{})
        for direction,d in case.get('directions',{}).items():
            if d.get('status')=='unavailable':lines.append(f"| {name} | {direction} unavailable: {d.get('reason')} | — | — | — | — | — | — | — |")
            for role,r in d.get('roles',{}).items():
                g=r.get('geometry',{}).get('whole',{})
                lines.append(f"| {name} | {direction} | {role} | "+' | '.join(fmt(g.get(k)) for k in ('direction_norm','energy_slope','force_slope','weighted_slope','energy_cosine','force_cosine'))+' |')
    lines+=['','| Case | Direction | Scale | Role | Raw E MSE | Raw F MSE | E original fraction | F original fraction | E source change % | F source change % |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        for p in cases.get(name,{}).get('probes',[]):
            for role in ('fitting','anchor'):
                r=p['roles'][role];m=p['metrics'][role];frac=r['original_baseline_fraction'];change=r['source_relative_change']
                lines.append(f"| {name} | {p['direction']} | {p['scale']} | {role} | "+' | '.join(fmt(v) for v in (m[KEYS[0]],m[KEYS[1]],frac[KEYS[0]],frac[KEYS[1]],100*change[KEYS[0]] if change[KEYS[0]] is not None else None,100*change[KEYS[1]] if change[KEYS[1]] is not None else None))+' |')
    lines+=['','## Verification, failures and limits','']
    for name in NAMES:
        c=cases.get(name,{})
        lines += [f"- {name}: {c.get('status','missing')}; attempted {bool(c.get('executed'))}; retained counts {counts(c)}; failure/stop: {c.get('failure_reason',c.get('stop_reason','none'))}."]
    if not compact['interpretation_valid']:lines+=['','Scientific interpretation blocked: source, execution, state, weighted/manual/builder identity or directional FD verification failed. Resolve the retained operational/verification failure before further comparison.',f"Root failure: {source.get('failure_reason','see case records')}."]
    else:lines+=['','These verified local slopes and fixed finite displacements describe the three saved states. Negative fitting force slope plus lower finite force indicates a feasible local improvement at that point. Positive slope or increase describes the exact saved direction/nonlinearity; one local step cannot establish accumulated optimizer causality. These observations do not establish convergence, capacity limits or eventual gate passage. No future training intervention has been executed.']
    lines+=['','Completed model/gradient frame passes by category: `'+json.dumps(compact['completed_frame_evaluations'],sort_keys=True)+'`. Opaque parent source verification and reference-builder work count only after synchronized completion; interrupted operations retain unknown partial work. These counts are diagnostic passes, not hardware FLOPs or training exposures.',
        '',f"Source summary SHA256 `{compact['source_summary_sha256']}`; report generator SHA256 `{compact['report_generator_sha256']}`.",
        'Complete raw/per-group/equal-group/overlap metrics, original fractions, weighted losses, both FD steps, block geometry, failed/partial work and all file checksums: [optimizer-step-diagnostics.json](optimizer-step-diagnostics.json). Trial trail: [PAPER_EVIDENCE_LOG.md](PAPER_EVIDENCE_LOG.md), OPTSTEP entries.','']
    for p in paths:lines += [f'![{p.stem}](figures/{p.name})','']
    (output/'optimizer-step-diagnostics.md').write_text('\n'.join(lines))
    return compact


if __name__=='__main__':generate(ROOT/'runs/aspirin-optimizer-step-2026-09-30',ROOT/'reports')
