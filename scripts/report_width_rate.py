"""Saved-byte-only width/rate reconciliation and balanced scientific report."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from force_consistency.width_rate_contract import (CONFIG_ID,MSE_KEYS,canonical_hash,digest,read,expected_updates,
    validate_contract,verify_width_rate_sources,resolve_commit)

ROLES=('fitting','anchor')
NAMES=('width16-lr1e-3','width16-lr3e-4','width32-lr1e-3','width32-lr3e-4')


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-8)

def normalized(value):return json.loads(json.dumps(value,allow_nan=False))


def gate(metrics,base,fraction):
    ratios={k:metrics[k]/base[k] for k in MSE_KEYS}
    return {'accepted':all(math.isfinite(v) and v<fraction for v in ratios.values()),'fraction_remaining':ratios,
            'threshold_mse':{k:base[k]*fraction for k in MSE_KEYS},'gate_fraction':fraction}


def _metrics_valid(metrics,inputs):
    for role in ROLES:
        m=metrics[role];meta=inputs[role];ids=meta['record_ids'];members=meta['memberships']
        if m['record_ids']!=ids or m['record_count']!=len(ids) or m['energy_count']!=len(ids):return False
        shape=meta['arrays']['forces']['shape'];components=math.prod(shape)
        if m['force_component_count']!=components:return False
        if any(not finite(m[k]) or m[k]<0 for k in MSE_KEYS):return False
        for k,rms in zip(MSE_KEYS,('energy_rmse_eV','force_rmse_eV_per_A')):
            if not close(m[rms],math.sqrt(m[k])):return False
        groups=sorted({g for r in ids for g in members[r]['groups']})
        if set(m['per_group'])!=set(groups):return False
        for group in groups:
            sub=m['per_group'][group];wanted=[r for r in ids if group in members[r]['groups']]
            if sub['record_ids']!=wanted or sub['record_count']!=len(wanted) or sub['energy_count']!=len(wanted) or sub['force_component_count']!=len(wanted)*math.prod(shape[1:]):return False
            if any(not finite(sub[k]) or sub[k]<0 for k in MSE_KEYS):return False
        if any(not close(m['equal_group'][k],sum(m['per_group'][g][k] for g in groups)/len(groups)) for k in MSE_KEYS):return False
    if not set(inputs['anchor']['record_ids'])<=set(inputs['fitting']['record_ids']):return False
    if metrics['anchor_fitted']!=metrics['anchor'] or metrics['anchor_fitted_flags']!={r:True for r in inputs['anchor']['record_ids']}:return False
    empty=metrics['anchor_not_fitted']
    return (empty['record_count']==empty['energy_count']==empty['force_component_count']==0 and empty['record_ids']==[] and empty['per_group']=={}
            and all(empty[k] is None and empty['equal_group'][k] is None for k in MSE_KEYS))


def verified_arm(arm,config,sources):
    try:
        validate_contract(config);case=arm['case'];prov=arm['provenance'];v=arm['checkpoint_verification'];oracle=arm['oracle_verification']
        if case not in config['cases'] or case!=prov['case'] or prov['config_identity']!=CONFIG_ID:return False
        if (arm['status']!='completed' or not arm['executed'] or not arm['execution_valid'] or arm['stop_reason']!='fixed_budget' or arm['full_pilot']!='pending'
            or not arm['source_hashes_unchanged'] or arm['pending_update_unknown'] or not arm['checkpoint_new_process_verified'] or not arm['oracle_verified']
            or not arm['warmup_source_unchanged'] or arm['warmup_work_status']!='completed' or arm['oracle_work_status']!='completed'):return False
        if (arm['updates'],arm['attempted_updates'],arm['persisted_update'],arm['frame_exposures'],arm['persisted_frame_exposures'],arm['warmup_frame_exposures'],
            arm['fitting_frame_count'],arm['anchor_frame_count'],arm['evaluation_passes'],arm['evaluation_frame_count'],arm['oracle_metrics_prediction_frames'],arm['reference_prediction_frames'])!=(6000,6000,6000,768000,768000,128,128,16,64,9216,144,144):return False
        norm=sources['normalization'];inputs=sources['historical_inputs'];base=sources['original_baselines']
        if (arm['original_baselines']!=base or prov['original_baselines']!=base or prov['normalization']!=norm or prov['inputs']!=inputs
            or prov['model']!={**config['model'],'features':case['features']} or prov['precision']!='float64' or prov['backend']!='cpu'
            or prov['source_pin_manifest_identity']!=config['source_pin_manifest_sha256'] or prov['sources']!=normalized(sources)):return False
        initial=arm['initial_state_hashes']
        if not initial or initial!=arm['initial_state_hashes_after'] or initial!=prov['initial_state_hashes'] or initial!=oracle['initial_state_hashes']:return False
        if case['features']==16 and arm['initial_raw_state_hashes']!=sources['historical_initial_state_hashes']:return False
        if not isinstance(arm['parameter_count'],int) or arm['parameter_count']<=0:return False
        if (oracle['status']!='passed' or oracle['count']!=1 or not oracle['discarded'] or oracle['atol']!=1e-10 or oracle['rtol']!=1e-8
            or oracle['fullbatch_reference_frames']!=128 or oracle['accumulated_proposal_frames']!=128 or oracle['prediction_reference_frames']!=256
            or not finite(oracle['state_maximum_absolute_difference']) or oracle['state_maximum_absolute_difference']<0
            or not finite(oracle['prediction_maximum_absolute_difference']) or oracle['prediction_maximum_absolute_difference']<0
            or len(oracle['gradient_sha256'])!=64
            or any(not finite(oracle[k]) or not 0<=oracle[k]<=1 for k in ('state_maximum_tolerance_fraction','prediction_maximum_tolerance_fraction'))):return False
        if (v['status']!='passed' or v['update']!=6000 or v['adam_count']!=6000 or v['normalization']!=norm or v['state_hashes']!=arm['final_state_hashes']
            or v['checkpoint_sha256']!=arm['checkpoint_sha256'] or len(v['checkpoint_sha256'])!=64 or len(v['history_sha256'])!=64
            or v['case']!=case or v['config_identity']!=CONFIG_ID or v['fitting_record_count']!=128 or v['anchor_record_count']!=16
            or v['precision']!='float64' or v['template_initializations']!=1 or v['template_initializer_frames']!=16 or v['prediction_frames']!=144
            or not finite(v['maximum_absolute_error']) or not 0<=v['maximum_absolute_error']<=1e-10 or v['tolerance']!=1e-10):return False
        elapsed,init,charged=arm['elapsed_seconds'],arm['shared_initialization_seconds'],arm['charged_elapsed_seconds']
        if any(not finite(x) or x<0 for x in (elapsed,init,charged)) or not close(charged,elapsed+init) or charged>=case['wall_seconds']:return False
        for key in ('compile_seconds','warmup_seconds','first_step_reference_seconds','probe_seconds','checkpoint_seconds','steady_update_seconds_total','steady_update_seconds_mean','evaluation_seconds_total'):
            if not finite(arm[key]) or arm[key]<0:return False
        if not close(arm['steady_update_seconds_total']/6000,arm['steady_update_seconds_mean']):return False
        h=arm['history']
        if [r['update'] for r in h]!=expected_updates():return False
        for row in h:
            if row['frame_exposures']!=row['update']*128 or not _metrics_valid(row['metrics'],inputs):return False
            for role,label in [('fitting','fit_gate'),('anchor','anchor_gate')]:
                expected=gate(row['metrics'][role],base[role],.1)
                if row[label]!=expected or row['original_baseline_fraction'][role]!=expected['fraction_remaining']:return False
                weighted=config['loss_weights']['energy']*row['metrics'][role][MSE_KEYS[0]]/norm['energy_loss_scale_eV']**2+config['loss_weights']['force']*row['metrics'][role][MSE_KEYS[1]]/norm['force_loss_scale_eV_per_A']**2
                if not close(row['weighted_loss'][role],weighted):return False
        if any(not close(h[1]['metrics'][r][k],arm['oracle_metrics'][r][k]) for r in ROLES for k in MSE_KEYS):return False
        wanted=list(sources['historical']) if case['name']=='width16-lr1e-3' else []
        if not arm['replay_verified'] or arm['replay_compared_updates']!=wanted:return False
        if wanted:
            for row in h:
                if row['update'] in sources['historical'] and any(not close(row['metrics'][r][k],sources['historical'][row['update']][r][k]) for r in ROLES for k in MSE_KEYS):return False
        return arm['final_metrics']==h[-1]['metrics'] and arm['fit_gate']==h[-1]['fit_gate'] and arm['anchor_gate']==h[-1]['anchor_gate']
    except (KeyError,TypeError,ValueError,ZeroDivisionError,OverflowError):return False


def factorial_contrasts(arms):
    unavailable={'available':False,'reason':'Requires four complete interpretation-valid cells and positive finite endpoints'}
    if set(arms)!=set(NAMES) or any(not a.get('interpretation_valid') or not a.get('execution_valid') or a.get('updates')!=6000 for a in arms.values()):return unavailable
    ratios={}
    try:
        for role in ROLES:
            ratios[role]={}
            for key in MSE_KEYS:
                x=[arms[n]['final_metrics'][role][key] for n in NAMES]
                if any(not finite(v) or v<=0 for v in x):return unavailable
                a,b,c,d=x
                ratios[role][key]={'rate_within_width16':b/a,'rate_within_width32':d/c,'width_at_rate1e-3':c/a,
                                  'width_at_rate3e-4':d/b,'interaction_ratio_of_rate_ratios':(d/c)/(b/a)}
        return {'available':True,'ratios':ratios,'interpretation':'One-seed fixed-support descriptive sensitivity, no uncertainty or causal claim'}
    except (KeyError,TypeError,ZeroDivisionError):return unavailable


def _saved_arm_valid(root,arm,c,sources):
    try:
        if not verified_arm(arm,c,sources):return False
        if read(root/'config.json')!=c or read(root/'case.json')!=arm['case'] or read(root/'provenance.json')!=arm['provenance']:return False
        receipt,history=resolve_commit(root)
        if history!=arm['history'] or receipt['update']!=6000 or receipt['state_hashes']!=arm['final_state_hashes'] or receipt['checkpoint_sha256']!=arm['checkpoint_sha256'] or receipt['history_sha256']!=arm['checkpoint_verification']['history_sha256']:return False
        if read(root/'checkpoint-verification.json')!=arm['checkpoint_verification']:return False
        if read(root/'oracle-verification.json')!={**arm['oracle_verification'],'metrics':arm['oracle_metrics']}:return False
        contract=read(root/'checkpoint-contract.json');prov=arm['provenance']
        if (contract['case']!=arm['case'] or contract['config_identity']!=CONFIG_ID or contract['update']!=6000 or contract['normalization']!=sources['normalization']
            or contract['commit']!=receipt or contract['state_hashes']!=arm['final_state_hashes'] or contract['inputs']!=sources['historical_inputs']
            or contract['input_files_sha256']!=prov['input_files_sha256'] or contract['reference_sha256']!=digest(root/'reference.npz')):return False
        return all(digest(root/f'{r}-inputs.npz')==prov['input_files_sha256'][r] for r in ROLES)
    except (OSError,KeyError,TypeError,ValueError):return False


def generate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);errors=[]
    c=read(root/'config.json');validate_contract(c);sources=verify_width_rate_sources(c)
    summary=read(root/'controller-failure.json') if (root/'controller-failure.json').exists() else read(root/'summary.json')
    if not (root/'sources.json').exists() or read(root/'sources.json')!=normalized(sources):errors.append('Saved source/control copies do not match pinned originals')
    arms={};verified={}
    for case in c['cases']:
        name=case['name'];child=root/name;original=summary.get('arms',{}).get(name,{'case':case,'status':'blocked','executed':False,'execution_valid':False})
        file=child/'failure.json' if (child/'failure.json').exists() else child/'summary.json'
        arm=read(file) if file.exists() else original
        valid=arm==original and _saved_arm_valid(child,arm,c,sources)
        arms[name]={**arm,'interpretation_valid':valid};verified[name]=valid
        if not valid:errors.append(name+': incomplete or invalid saved execution/ancestry')
    matrix_valid=all(verified.values()) and summary.get('execution_valid') and summary.get('status')=='completed' and summary.get('full_pilot')=='pending'
    elapsed=summary.get('elapsed_seconds')
    if not finite(elapsed) or not 0<=elapsed<c['total_wall_seconds']:matrix_valid=False;errors.append('Global budget missing or exceeded')
    try:
        parent=summary['parent_work'];shared=summary['shared_initializations']
        if parent['status']!='completed' or parent['completed_source_prediction_frames']!=208 or parent['historical_template_initializations']!=1:raise ValueError('Parent work mismatch')
        if set(shared)!={'16','32'}:raise ValueError('Shared initialization count mismatch')
        for width in (16,32):
            init=shared[str(width)];paired=[arms[f'width{width}-lr1e-3'],arms[f'width{width}-lr3e-4']]
            if (init['status']!='completed' or init['features']!=width or init['completed_initializations']!=1 or init['baseline_prediction_frames']!=144
                or any(a['initial_state_hashes']!=init['initial_state_hashes'] or a['parameter_count']!=init['parameter_count']
                       or not close(a['shared_initialization_seconds'],init['elapsed_seconds']) for a in paired)):raise ValueError('Paired initialization identity/accounting mismatch')
        if shared['32']['parameter_count']<=shared['16']['parameter_count']:raise ValueError('Width parameter count mismatch')
    except (KeyError,TypeError,ValueError):matrix_valid=False;errors.append('Shared/parent initialization provenance is incomplete or inconsistent')
    if errors:matrix_valid=False
    contrast_arms=arms if matrix_valid else {n:{**a,'interpretation_valid':False} for n,a in arms.items()}
    contrasts=factorial_contrasts(contrast_arms)
    terminal={}
    for name,arm in arms.items():
        rows={r['update']:r for r in arm.get('history',[])}
        if verified[name] and 5500 in rows:
            terminal[name]={role:{key:{'relative_change':(rows[6000]['metrics'][role][key]-rows[5500]['metrics'][role][key])/rows[5500]['metrics'][role][key],
                               'below_one_percent':abs((rows[6000]['metrics'][role][key]-rows[5500]['metrics'][role][key])/rows[5500]['metrics'][role][key])<.01}
                               for key in MSE_KEYS if rows[5500]['metrics'][role][key]>0} for role in ROLES}
    counts={'synchronized_updates':sum(a.get('updates',0) for a in arms.values()),'attempted_updates':sum(a.get('attempted_updates',0) for a in arms.values()),
            'committed_updates':sum(a.get('persisted_update') or 0 for a in arms.values()),'committed_frame_exposures':sum(a.get('persisted_frame_exposures',0) for a in arms.values()),
            'synchronized_frame_exposures':sum(a.get('frame_exposures',0) for a in arms.values()),'discarded_warmup_frames':sum(a.get('warmup_frame_exposures',0) for a in arms.values()),
            'unknown_pending_cells':[n for n,a in arms.items() if a.get('pending_update_unknown') or a.get('oracle_work_status')=='unknown' or a.get('warmup_work_status')=='unknown'],
            'shared_training_initializations_completed':sum(v.get('completed_initializations') or 0 for v in summary.get('shared_initializations',{}).values()),
            'parent_work':summary.get('parent_work',summary.get('preflight_work'))}
    artifacts={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    result={'status':summary.get('status'),'interpretation_valid':bool(matrix_valid),'execution_valid':summary.get('execution_valid',False),'full_pilot':'pending',
            'config':c,'config_identity':CONFIG_ID,'sources':sources,'arms':arms,'arm_verification':verified,'contrasts':contrasts,'terminal_window':terminal,
            'actual_counts':counts,'actual_matrix_elapsed_seconds':elapsed,'shared_initializations':summary.get('shared_initializations'),
            'artifacts_sha256':artifacts,'source_root':str(root),'report_generator_sha256':digest(__file__),'verification_errors':errors,
            'limits':['One seed; fully fitted anchors; no generalization, convergence, capacity or optimizer-causal conclusion.',
                      'A worse lower fresh fixed-rate endpoint does not test smaller late-stage rates or decay schedules.',
                      'Sequential wall times depend on order, compiler cache and system load; no architecture-efficiency ranking.',
                      'Width changes initialization shape, parameter count and optimization geometry together.',
                      'Original strict joint gates are unchanged. No automatic winner, candidate or full-pilot clearance.',
                      'Boundary caps cannot forcibly cancel in-flight work; incomplete cells retain data, contrasts unavailable.']}
    (output/'width-rate-diagnostics.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Fresh width and fixed-rate diagnostic','',f"Saved execution status: **{result['status']}**; interpretation valid: **{result['interpretation_valid']}**. Full pilot: **pending**.",'',
           '| Cell | Saved updates | Persisted updates | Raw fitted E MSE | Raw fitted F MSE | Joint fitting gate | Joint anchor gate | Verified |',
           '|---|---:|---:|---:|---:|---|---|---|']
    for name,a in arms.items():
        metric=a.get('final_metrics',{}).get('fitting',{})
        lines.append(f"| {name} | {a.get('updates','unavailable')} | {a.get('persisted_update','unavailable')} | {metric.get(MSE_KEYS[0],'unavailable')} | {metric.get(MSE_KEYS[1],'unavailable')} | {a.get('fit_gate',{}).get('accepted','unavailable')} | {a.get('anchor_gate',{}).get('accepted','unavailable')} | {verified[name]} |")
    lines+=['',f"Actual counters: `{json.dumps(counts,sort_keys=True)}`.",'',f"Measured matrix elapsed: {elapsed} seconds. Shared training initialization is actual work once per width; paired cap charges are separate. Historical parent template initialization/source verification and discarded reference/warmup work are retained separately.",'',
            f"Four-cell endpoint contrasts available: **{contrasts['available']}**. Ratios and descriptive 5500→6000 window changes are in the JSON; no selected endpoint.",'',
            'Every raw/per-group/equal-group/overlap history, original fraction, weighted loss, checkpoint pair, receipt, parameter count and timing is preserved in the JSON and raw artifacts. Learning failure is distinct from execution failure; all attempts remain in the denominators.','',
            '![Every prescribed raw measurement and original gate](figures/width-rate-curves.png)','']
    lines+=['- '+limit for limit in result['limits']]
    if errors:lines+=['','Verification issues:']+['- '+error for error in errors]
    (output/'width-rate-diagnostics.md').write_text('\n'.join(lines)+'\n')
    _figure(arms,sources,output/'figures/width-rate-curves.png')
    return result


def _figure(arms,sources,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for i,role in enumerate(ROLES):
        for j,key in enumerate(MSE_KEYS):
            ax=axes[i,j]
            for name,arm in arms.items():
                rows=[r for r in arm.get('history',[]) if role in r.get('metrics',{}) and finite(r['metrics'][role].get(key))]
                if rows:ax.plot([r['update'] for r in rows],[r['metrics'][role][key] for r in rows],marker='.',markersize=3,label=name+(' [partial/invalid]' if not arm['interpretation_valid'] else ''))
            ax.axhline(.1*sources['original_baselines'][role][key],color='black',linestyle='--',label='Original strict gate')
            ax.set_yscale('log');ax.set_xlabel('Fresh committed update (prescribed observations)');ax.set_ylabel('Raw energy MSE (eV²)' if j==0 else 'Raw force MSE ((eV/Å)²)')
            ax.set_title(role+(' energy' if j==0 else ' forces'));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('One seed, fixed fitting support; fully fitted anchors; full pilot pending')
    fig.savefig(path,dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('reports'));args=parser.parse_args()
    result=generate(args.root,args.output);print(json.dumps({'interpretation_valid':result['interpretation_valid'],'contrasts_available':result['contrasts']['available'],'full_pilot':'pending'},indent=2))
    raise SystemExit(0 if result['interpretation_valid'] else 1)
