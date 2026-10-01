"""Pure saved-record reconciliation for the fresh replacement. No training or array loads."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from force_consistency.width_rate_contract import MSE_KEYS,canonical_hash,digest,read,expected_updates,resolve_commit
from force_consistency.width_rate_replacement_contract import CONFIG_ID,validate_replacement,verify_replacement_sources
from force_consistency.width_rate_replacement import _audit_cell,audit_replacement_cell,valid_child,attempt_binding
_spec=importlib.util.spec_from_file_location('pinned_width_rate_report',Path(__file__).with_name('report_width_rate.py'))
_old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_old)
# Only these configuration-independent helpers cross the old report boundary.
finite,close,normalized,gate=_old.finite,_old.close,_old.normalized,_old.gate
_metrics_valid,factorial_contrasts,preflight_accounting=_old._metrics_valid,_old.factorial_contrasts,_old.preflight_accounting
ROLES=('fitting','anchor')
NAMES=('width16-lr1e-3','width16-lr3e-4','width32-lr1e-3','width32-lr3e-4')

def verified_arm(arm,config,sources):
    try:
        validate_replacement(config);case=arm['case'];prov=arm['provenance'];v=arm['checkpoint_verification'];oracle=arm['oracle_verification']
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
        if any(not close(h[0]['metrics'][r][k],base[r][k]) for r in ROLES for k in MSE_KEYS):return False
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


def strict_read(path):
    def invalid(value):raise ValueError('Nonfinite saved JSON: '+value)
    value=json.loads(Path(path).read_text(),parse_constant=invalid)
    def check(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('Nonfinite saved JSON')
        if isinstance(v,dict):
            for x in v.values():check(x)
        elif isinstance(v,list):
            for x in v:check(x)
    check(value);return value


def optional(path,errors):
    try:return strict_read(path)
    except (OSError,ValueError,TypeError) as exc:
        errors.append(str(path)+': '+type(exc).__name__+': '+str(exc));return None


def _saved_arm_valid(child,arm,c,sources):
    try:
        if not verified_arm(arm,c,sources):return False
        audit=audit_replacement_cell(c,child);receipt=audit['commit'];history=audit['history']
        if (arm['provenance']!=audit['provenance'] or history!=arm['history'] or receipt['state_hashes']!=arm['final_state_hashes']
            or receipt['checkpoint_sha256']!=arm['checkpoint_sha256'] or read(child/'checkpoint-verification.json')!=arm['checkpoint_verification']
            or read(child/'oracle-verification.json')!={**arm['oracle_verification'],'metrics':arm['oracle_metrics']}):return False
        binding=attempt_binding(child,audit);attempt=read(child/'probe-attempt.json');entry=read(child/'probe-entry.json');completion=read(child/'probe-completion.json')
        if attempt!=binding or entry['binding']!=binding or entry['attempt_sha256']!=digest(child/'probe-attempt.json'):
            return False
        if completion['status']!='passed' or completion['attempt_sha256']!=entry['attempt_sha256'] or completion['verification_sha256']!=digest(child/'checkpoint-verification.json'):return False
        valid_child(audit,arm['checkpoint_verification'])
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def committed_observations(child,c,sources,errors):
    """Only the last published commit can establish observed persisted work."""
    if not (child/'commit.json').exists():return None,[]
    try:
        receipt,history=resolve_commit(child);prov=strict_read(child/'provenance.json');case=strict_read(child/'case.json')
        if (strict_read(child/'config.json')!=c or case not in c['cases'] or prov['case']!=case or prov['config_identity']!=CONFIG_ID
            or prov['inputs']!=sources['historical_inputs'] or prov['normalization']!=sources['normalization']
            or prov['original_baselines']!=sources['original_baselines']):raise ValueError('Published commit provenance invalid')
        updates=[r['update'] for r in history]
        if updates!=expected_updates()[:len(updates)] or receipt['update'] not in expected_updates():raise ValueError('Published history schedule invalid')
        for i,row in enumerate(history):
            if strict_read(child/f"histories/{row['update']}.json")!=history[:i+1]:raise ValueError('Published history prefix invalid')
            if row['frame_exposures']!=row['update']*128 or not _metrics_valid(row['metrics'],sources['historical_inputs']):raise ValueError('Published metrics topology invalid')
            for role,label in [('fitting','fit_gate'),('anchor','anchor_gate')]:
                expected=gate(row['metrics'][role],sources['original_baselines'][role],.1)
                if row[label]!=expected or row['original_baseline_fraction'][role]!=expected['fraction_remaining']:raise ValueError('Published gate rebase')
                if row['update']==0 and any(not close(row['metrics'][role][key],sources['original_baselines'][role][key]) for key in MSE_KEYS):raise ValueError('Published initial baseline changed')
        return receipt,history
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:
        errors.append(str(child)+': committed observations unavailable: '+str(exc));return None,[]


def _global_valid(root,c,sources,summary,arms,accounting,errors):
    try:
        if (not all(a['interpretation_valid'] for a in arms.values()) or summary['status']!='completed' or not summary['execution_valid']
            or summary['full_pilot']!='pending' or strict_read(root/'sources.json')!=normalized(sources)):raise ValueError('Matrix/arm/source state incomplete')
        parent=summary['parent_work'];shared=summary['shared_initializations'];elapsed=summary['elapsed_seconds']
        if accounting['parent_work']!=parent or accounting['shared_initializations']!=shared:raise ValueError('Preflight receipt mismatch')
        if not finite(elapsed) or not 0<=elapsed<c['total_wall_seconds']:raise ValueError('Global cap exceeded')
        if parent['status']!='completed' or parent['completed_source_prediction_frames']!=208 or parent['historical_template_initializations']!=1:raise ValueError('Parent work invalid')
        if not finite(parent['elapsed_seconds']) or parent['elapsed_seconds']<0 or set(shared)!={'16','32'}:raise ValueError('Shared time/count invalid')
        for width in (16,32):
            init=shared[str(width)];paired=[arms[f'width{width}-{rate}'] for rate in ('lr1e-3','lr3e-4')]
            if (init['status']!='completed' or init['features']!=width or init['completed_initializations']!=1 or init['baseline_prediction_frames']!=144
                or init['initialization_batch_record_ids']!=sources['historical_inputs']['anchor']['record_ids'][:16]
                or not finite(init['elapsed_seconds']) or init['elapsed_seconds']<0
                or any(not close(init['initial_metrics'][r][k],sources['original_baselines'][r][k]) for r in ROLES for k in MSE_KEYS)
                or any(a['initial_state_hashes']!=init['initial_state_hashes'] or a['parameter_count']!=init['parameter_count'] or not close(a['shared_initialization_seconds'],init['elapsed_seconds']) for a in paired)):
                raise ValueError('Paired fresh initializer identity/baseline mismatch')
        if shared['32']['parameter_count']<=shared['16']['parameter_count']:raise ValueError('Width parameter count invalid')
        minimum=sum(a['elapsed_seconds'] for a in arms.values())+sum(e['elapsed_seconds'] for e in shared.values())+parent['elapsed_seconds']
        if elapsed+1e-6<minimum:raise ValueError('Sequential timing contradiction')
        completion=strict_read(root/'controller-completion.json');invocation=strict_read(root/'invocation.json')
        if completion['summary_sha256']!=digest(root/'summary.json') or completion['status']!='completed' or not finite(completion['elapsed_seconds']) or not elapsed<=completion['elapsed_seconds']<c['total_wall_seconds']:
            raise ValueError('Final publication receipt invalid')
        if invocation['ordinal']!=1 or invocation['config_identity']!=CONFIG_ID or invocation['role']!=c['replacement_role']:raise ValueError('Invocation identity invalid')
        return True
    except (OSError,ValueError,KeyError,TypeError) as exc:errors.append('Global validity: '+str(exc));return False


def generate(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);errors=[]
    c=strict_read(root/'config.json');validate_replacement(c);sources=verify_replacement_sources(c)
    terminal=root/'controller-failure.json' if (root/'controller-failure.json').exists() else root/'summary.json'
    summary=optional(terminal,errors) or {};arms={};valid={};committed=0;unknown=[]
    for case in c['cases']:
        name=case['name'];child=root/name;raw_summary=summary.get('arms',{}).get(name)
        file=child/'failure.json' if (child/'failure.json').exists() else child/'summary.json'
        saved=optional(file,errors) if file.exists() else None
        raw=saved if saved is not None else raw_summary
        arm=raw if isinstance(raw,dict) else {'case':case,'status':'unavailable','executed':None,'execution_valid':False,'stop_reason':'missing_terminal_record'}
        is_valid=arm==raw_summary and _saved_arm_valid(child,arm,c,sources)
        receipt,history=committed_observations(child,c,sources,errors)
        if receipt:committed+=receipt['update']
        # Failed/partial counters stay raw claims; only independent commit/child bindings become known work.
        child_work=None;child_result=None
        try:
            audit=audit_replacement_cell(c,child);child_result=strict_read(child/'checkpoint-verification.json');child_work=valid_child(audit,child_result)
        except (OSError,ValueError,KeyError,TypeError):
            if arm.get('executed') is False:child_work={'template_initializations':0,'template_initializer_frames':0,'prediction_frames':0,'training_updates':0}
        if not is_valid:unknown.append(name);errors.append(name+': incomplete or invalid saved execution')
        arms[name]={**arm,'execution_valid':bool(is_valid),'interpretation_valid':bool(is_valid),'raw_terminal_claims':arm,
                    'history':history,'persisted_update':receipt['update'] if receipt else None,
                    'derived_committed_frame_exposures':receipt['update']*128 if receipt else 0,
                    'derived_synchronized_updates':arm['updates'] if is_valid else 0 if arm.get('executed') is False else None,
                    'derived_attempted_updates':arm['attempted_updates'] if is_valid else 0 if arm.get('executed') is False else None,
                    'known_child_work':child_work,'known_child_receipt':child_result,'observation_status':'complete' if is_valid else 'committed_prefix' if history else 'unavailable'}
        if history:
            arms[name].update(final_metrics=history[-1]['metrics'],fit_gate=history[-1]['fit_gate'],anchor_gate=history[-1]['anchor_gate'])
        else:
            for key in ('final_metrics','fit_gate','anchor_gate'):arms[name].pop(key,None)
        valid[name]=is_valid
    events=optional(root/'preflight-work.json',errors) or summary.get('preflight_work',[])
    accounting=preflight_accounting(summary,events);matrix_valid=_global_valid(root,c,sources,summary,arms,accounting,errors)
    raw_elapsed=summary.get('elapsed_seconds');elapsed=None
    try:
        completion=strict_read(root/'controller-completion.json')
        if (completion['summary_sha256']==digest(terminal) and completion['status']==summary.get('status')
            and finite(raw_elapsed) and raw_elapsed>=0 and finite(completion['elapsed_seconds']) and completion['elapsed_seconds']>=raw_elapsed):
            elapsed=raw_elapsed
    except (OSError,ValueError,KeyError,TypeError):pass
    if matrix_valid:elapsed=raw_elapsed
    counts={'committed_updates':committed,'committed_frame_exposures':committed*128,
        'synchronized_updates':sum(a['derived_synchronized_updates'] for a in arms.values()) if all(a['derived_synchronized_updates'] is not None for a in arms.values()) else None,
        'attempted_updates':sum(a['derived_attempted_updates'] for a in arms.values()) if all(a['derived_attempted_updates'] is not None for a in arms.values()) else None,
        'unknown_or_invalid_cell_counts':unknown,'known_probe_prediction_frames':sum((a['known_child_work'] or {}).get('prediction_frames',0) for a in arms.values()),
        'unknown_probe_cells':[n for n,a in arms.items() if a['known_child_work'] is None],**accounting}
    counts['work_counts_complete']=matrix_valid and accounting['work_counts_complete']
    contrast_arms=arms if matrix_valid else {n:{**a,'interpretation_valid':False} for n,a in arms.items()}
    terminal_window={}
    for name,arm in arms.items():
        rows={r['update']:r for r in arm['history']}
        if arm['interpretation_valid'] and 5500 in rows:
            terminal_window[name]={role:{k:{'relative_change':(rows[6000]['metrics'][role][k]-rows[5500]['metrics'][role][k])/rows[5500]['metrics'][role][k],
                'below_one_percent':abs((rows[6000]['metrics'][role][k]-rows[5500]['metrics'][role][k])/rows[5500]['metrics'][role][k])<.01}
                for k in MSE_KEYS if rows[5500]['metrics'][role][k]>0} for role in ROLES}
    result={'status':summary.get('status','unavailable'),'execution_valid':bool(matrix_valid),'interpretation_valid':bool(matrix_valid),'full_pilot':'pending',
        'config':c,'config_identity':CONFIG_ID,'sources':sources,'source_root':str(root),'arms':arms,'arm_verification':valid,
        'contrasts':factorial_contrasts(contrast_arms),'terminal_window':terminal_window,'actual_counts':counts,'actual_matrix_elapsed_seconds':elapsed,'raw_terminal_elapsed_seconds':raw_elapsed,
        'ancestry':{'original_failed':strict_read('reports/width-rate-diagnostics.json'),
                    'copy_verified':strict_read('reports/width-rate-recovery-verification.json'),'combined_work_counts_complete':False},
        'artifacts_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
        'report_generator_sha256':digest(__file__),'verification_errors':errors,
        'limits':['One seed and fully fitted anchors; no held-out/generalization, capacity or optimizer-causal conclusion.',
            'Fresh fixed-rate endpoints do not test late-stage rate decay. Width changes initialization shape and optimization geometry.',
            'Sequential times depend on order, compiler cache and load; no efficiency ranking.',
            'Original failed trial and zero-update copy stay separate; original unreceipted child effort remains unknown.',
            'Partial terminal counts are raw claims; known persisted observations and child work require independent saved bindings.',
            'No automatic winner or full-pilot clearance. All failures and blocked cases stay indexed.']}
    (output/'width-rate-replacement-diagnostics.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# Fresh replacement width/rate diagnostic','',f"Status: {result['status']}; interpretation valid: {matrix_valid}; full pilot: pending.",'',
        '| Cell | Status | Observed persisted update | Fitting E fraction | Fitting F fraction | Fitting joint gate | Anchor joint gate |',
        '|---|---|---:|---:|---:|---|---|']
    for name,a in arms.items():
        g=a.get('fit_gate',{});fractions=g.get('fraction_remaining',{})
        lines.append(f"| {name} | {a['status']} | {a['persisted_update']} | {fractions.get(MSE_KEYS[0])} | {fractions.get(MSE_KEYS[1])} | {g.get('accepted')} | {a.get('anchor_gate',{}).get('accepted')} |")
    lines+=['',f"Four-cell contrasts available: {result['contrasts']['available']}.",f"Measured elapsed seconds: {elapsed}.",'',
            'Every raw/per-group/equal-group/overlap observation, objective, gate, counter, phase time and immutable source identity is retained in the JSON and raw files.',
            '![Saved prescribed observations](figures/width-rate-replacement-curves.png)','']+['- '+x for x in result['limits']]
    if errors:lines+=['','Verification issues:']+['- '+x for x in errors]
    (output/'width-rate-replacement-diagnostics.md').write_text('\n'.join(lines)+'\n')
    _figure(arms,sources,output/'figures/width-rate-replacement-curves.png')
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
                rows=[r for r in arm.get('history',[]) if finite(r.get('metrics',{}).get(role,{}).get(key))]
                if rows:ax.plot([r['update'] for r in rows],[r['metrics'][role][key] for r in rows],marker='.',markersize=3,label=name+(' [committed partial/invalid]' if not arm['interpretation_valid'] else ''))
            ax.axhline(.1*sources['original_baselines'][role][key],color='black',linestyle='--',label='Original strict gate')
            ax.set_yscale('log');ax.set_xlabel('Observed prescribed update from published commit');ax.set_ylabel('Raw energy MSE (eV²)' if j==0 else 'Raw force MSE ((eV/Å)²)')
            ax.set_title(role+(' energy' if j==0 else ' forces'));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Fresh replacement: one seed, fully fitted anchors; pilot pending')
    fig.savefig(path,dpi=170);plt.close(fig)


def reconcile_evidence(result,evidence_path,artifacts):
    errors=[]
    if artifacts.get('stage')!='final':return {'stage':'preliminary','log_completeness':False,'pending':['deferred_review','preservation','navigation_or_blocked','results_commit','final_cutoff'],'errors':[]}
    evidence=Path(evidence_path).read_bytes();n=artifacts.get('evidence_prefix_bytes');want=artifacts.get('evidence_prefix_sha256')
    if not isinstance(n,int) or n<0 or n>len(evidence) or hashlib.sha256(evidence[:n]).hexdigest()!=want:
        errors.append('Evidence prefix changed or missing');prefix=''
    else:prefix=evidence[:n].decode()
    required=artifacts.get('required_artifacts',{});roles=artifacts.get('required_roles',{})
    for role in ('deferred_review','preservation','navigation','results_commit','code','config','pins','raw','reports','tests','plan_review','engineering_logs','ancestry'):
        if roles.get(role) not in required:errors.append('Required later artifact omitted: '+role)
    for name,want in required.items():
        try:
            if digest(name)!=want:errors.append('Required artifact changed: '+name)
        except OSError:errors.append('Required artifact missing: '+name)
    if 'source_root' in result:
        for name,want in result.get('artifacts_sha256',{}).items():
            full=str(Path(result['source_root'])/name)
            if required.get(full)!=want:errors.append('Raw artifact coverage omitted or changed: '+full)
    if 'sources' in result:
        for name,want in result['sources'].get('pins',{}).items():
            if required.get(name)!=want:errors.append('Immutable ancestor/source coverage omitted: '+name)
    cases=artifacts.get('case_evidence',{})
    if set(cases)!=set(result['arms']):errors.append('Case evidence omitted or substituted')
    for name,arm in result['arms'].items():
        marker=cases.get(name)
        if not isinstance(marker,str) or marker not in prefix:errors.append('Case outcome omitted: '+name)
    for marker in artifacts.get('required_evidence',[]):
        if marker not in prefix:errors.append('Required trial/result evidence missing: '+marker)
    exclusions=artifacts.get('exclusions',[])
    if set(exclusions)!={'coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha'}:errors.append('Acyclic cutoff exclusions incomplete')
    return {'stage':'final','log_completeness':not errors,'errors':errors,'scope':'Named replacement, original failed and separate copy milestones; artifact/log coverage, not a rerun of all historical science.',
            'evidence_prefix_bytes':n,'evidence_prefix_sha256':artifacts.get('evidence_prefix_sha256'),'required_artifact_count':len(required),'exclusions':exclusions}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('reports'));args=parser.parse_args()
    result=generate(args.root,args.output);print(json.dumps({'status':result['status'],'interpretation_valid':result['interpretation_valid'],'full_pilot':'pending'}))
    raise SystemExit(0 if result['interpretation_valid'] else 1)
