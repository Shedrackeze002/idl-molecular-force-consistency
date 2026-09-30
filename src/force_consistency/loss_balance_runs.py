"""Frozen calibration and ordered, matched-budget loss controls with preserved failures."""
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import numpy as np
from .coverage import take_subset
from .coverage_matrix import _blocked,_failed_arm,_retain_or_block,append_coverage_evidence
from .coverage_runs import MSE_KEYS,run_coverage_arm
from .data.prepare import write_json
from .data.splits import canonical_hash
from .diagnostics import sha256
from .gradient_runs import CASE_NAMES,load_gradient_sources,verify_source_hashes
from .loss_balance import validate_weights,build_weighted_step
from .runs import code_provenance

ARMS=('replay','balanced','force-only')
CALIBRATION_FILES={'config.json','provenance.json','subsets.json','phase-A.json','summary.json','cases.json'}|{
    f'{case}/{name}' for case in CASE_NAMES for name in ('prospective.json','partial.json','observations.json','summary.json')}


def _read(path):return json.loads(Path(path).read_text())


def verify_calibration_hashes(config):
    if sha256(config['calibration_path'])!=config['calibration_sha256']:raise ValueError('Calibration compact identity mismatch')
    compact=_read(config['calibration_path']);root=Path(config['calibration_run'])
    if set(compact['sources'])!=CALIBRATION_FILES:raise ValueError('Incomplete calibration raw allowlist')
    for file,digest in compact['sources'].items():
        if sha256(root/file)!=digest:raise ValueError(f'Calibration raw identity mismatch: {file}')
    return compact


def balance_preflight(config):
    verify_source_hashes(config)
    compact=verify_calibration_hashes(config);root=Path(config['calibration_run'])
    raw=_read(root/'fresh-C-fitting/observations.json');case=compact['cases'][config['calibration_case']]
    if config['calibration_case']!='fresh-C-fitting':raise ValueError('Calibration case mismatch')
    for key in ('record_count','force_component_count','update','geometry','losses','state_hashes_before','state_hashes_after','verification_valid'):
        if raw[key]!=case[key]:raise ValueError(f'Calibration raw/compact mismatch: {key}')
    if raw['record_count']!=128 or raw['force_component_count']!=8064 or raw['update']!=0 or not raw['verification_valid'] or raw['state_hashes_before']!=raw['state_hashes_after']:
        raise ValueError('Invalid fresh calibration')
    if not compact['execution']['execution_valid'] or not _read(root/'summary.json')['execution_valid'] or case['status']!='completed':
        raise ValueError('Invalid calibration execution')
    g=raw['geometry']['whole'];e,f=g['energy_norm'],g['force_norm']
    if not np.isfinite([e,f]).all() or min(e,f)<=0:raise ValueError('Invalid calibration norms')
    ratio=e/f;expected={'replay':{'energy':1.,'force':1.},'balanced':{'energy':2/(1+ratio),'force':2*ratio/(1+ratio)},'force-only':{'energy':0.,'force':2.}}
    if (config['calibration_energy_norm'],config['calibration_force_norm'],config['calibration_ratio'])!=(e,f,ratio) or config['arms']!=expected:
        raise ValueError('Frozen calibration formula/weights mismatch')
    for weights in config['arms'].values():validate_weights(weights)
    c=_read(root/'config.json');subsets=_read(root/'subsets.json');covsubsets=_read(Path(config['source_run'])/'subsets.json')
    if subsets!=covsubsets or raw['record_ids']!=subsets['subsets']['diverse128']['record_ids'] or c['source_sha256']!=config['source_sha256']:
        raise ValueError('Calibration subset/source metadata mismatch')
    for key in ('expected_prepared_identity','expected_split_hash','expected_subsets_identity','seed','model','microbatch_size','learning_rate','backend'):
        if c[key]!=config[key]:raise ValueError(f'Calibration setting mismatch: {key}')
    if (config['max_updates'],config['validation_interval'],config['gate_fraction'],config['wall_seconds'])!=(2000,100,.1,1800):
        raise ValueError('Frozen budget/gate mismatch')
    if config.get('resume_checkpoint') or config.get('early_stopping'):raise ValueError('No resume/early stop allowed')
    # All source/calibration identities/formula gates above precede data access.
    train,manifest,norm,selected,fresh,final,historical=load_gradient_sources(config)
    if raw['state_hashes_before']!=historical['verification']['initial_state_hashes'] or _read(root/'provenance.json')['normalization']!=asdict(norm):
        raise ValueError('Calibration fresh-state/normalization mismatch')
    initial=historical['arms']['diverse128']['summary']['initial_metrics']['fitting']
    for objective,key,scale in [('energy',MSE_KEYS[0],norm.energy_loss_scale_eV),('force',MSE_KEYS[1],norm.force_loss_scale_eV_per_A)]:
        if not np.isclose(raw['losses'][objective],initial[key]/scale**2,atol=1e-10,rtol=1e-8):raise ValueError('Calibration initial loss mismatch')
    return train,manifest,norm,selected,historical,{'raw':raw,'compact':compact,'ratio':ratio}


def verify_balance_replay(current,source,tolerance):
    scheduled=[0,*range(100,2001,100),250];scheduled=sorted(scheduled)
    for arm in (current,source):
        if not arm['execution_valid'] or arm['updates']!=2000 or not arm['checkpoint_new_process_verified']:
            raise ValueError('Replay source/current endpoint invalid')
        updates=[r['update'] for r in arm['history']]
        if updates!=scheduled:raise ValueError('Replay schedule missing/duplicate/out of order')
    if current['initial_state_hashes']!=source['initial_state_hashes']:raise ValueError('Replay initialization mismatch')
    maximum=0.
    for actual,want in zip(current['history'],source['history'],strict=True):
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:
                a,b=actual['metrics'][role][key],want['metrics'][role][key]
                if not np.isfinite([a,b]).all() or not np.isclose(a,b,**tolerance):raise ValueError(f'Replay mismatch: {actual["update"]}/{role}/{key}')
                maximum=max(maximum,abs(a-b))
    return {'status':'passed','compared_updates':scheduled,'maximum_absolute_mse_difference':maximum,'tolerance':tolerance}


def execute_balance_matrix(root,runner,replay):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ARMS):raise ValueError('Arm directory already exists')
    result={'status':'failed','execution_valid':False,'full_pilot':'pending','balanced_joint_accepted':False,'arms':{}}
    blocked=None;expected=None
    for name in ARMS:
        if blocked:
            result['arms'][name]=_blocked(root,name,blocked);write_json(root/'summary.json',result);continue
        try:
            arm=runner(name,expected);result['arms'][name]=arm
            if not arm['execution_valid']:blocked=f'{name}_operational_or_budget_failure'
            elif name=='replay':
                try:result['replay']=replay(arm)
                except Exception as exc:
                    result['replay']={'status':'failed','failure_reason':f'{type(exc).__name__}: {exc}'}
                    write_json(root/name/'replay-comparison-failure.json',result['replay']);raise
                expected=arm['initial_state_hashes']
        except Exception as exc:
            result['failure_reason']=f'{type(exc).__name__}: {exc}';blocked=f'{name}_execution_or_replay_failure'
            if name not in result['arms']:result['arms'][name]=_failed_arm(root,name,exc)
        write_json(root/'summary.json',result)
    result['execution_valid']=blocked is None;result['status']='completed' if result['execution_valid'] else 'failed'
    candidate=result['arms']['balanced'];result['balanced_joint_accepted']=bool(result['execution_valid'] and candidate['fit_gate']['accepted'] and candidate['anchor_gate']['accepted'])
    forceonly=result['arms']['force-only']
    result['force_only_force_accepted']=bool(result['execution_valid'] and all(
        forceonly[gate].get('fraction_remaining',{}).get(MSE_KEYS[1],float('inf'))<forceonly[gate].get('gate_fraction',.1)
        for gate in ('fit_gate','anchor_gate')))
    result['next_step']='resolve_execution_failure' if not result['execution_valid'] else ('separately_review_pilot_amendment' if result['balanced_joint_accepted'] else 'review_fixed_loss_outcomes_before_new_intervention')
    write_json(root/'summary.json',result);return result


def diagnose_loss_balance(config):
    root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',config);stage='preflight'
    append_coverage_evidence(config,'BALANCE-EXECUTION-PRE','Frozen fixed loss controls preregistered',{'config':config,'config_hash':canonical_hash(config),'limits':'One exact3armmatrix; no sweep/retry/adaptation; training-only fitted anchor'})
    try:
        train,manifest,norm,selected,historical,calibration=balance_preflight(config)
        fitting=take_subset(train,selected['subsets']['diverse128']['record_ids']);anchor=take_subset(train,selected['subsets']['narrow16']['record_ids'])
        source=historical['arms']['diverse128']['summary'];expected=historical['verification']['initial_state_hashes']
        provenance={**code_provenance(Path(__file__).resolve().parents[2]),'dataset_identity':manifest['identity'],'split_hash':manifest['split_hash'],
            'frozen_subsets_identity':config['expected_subsets_identity'],'normalization':asdict(norm),'precision':'float64','backend':'cpu',
            'historical_reference':historical,'calibration_identity':config['calibration_sha256'],'calibration_raw_sources':calibration['compact']['sources'],
            'calibration_ratio_energy_over_force':calibration['ratio'],'expected_initial_hashes':expected,
            'fitting_initial_mse':{k:source['initial_metrics']['fitting'][k] for k in MSE_KEYS},
            'anchor_initial_mse':{k:source['initial_metrics']['anchor'][k] for k in MSE_KEYS}}
        write_json(root/'provenance.json',provenance);write_json(root/'subsets.json',selected);stage='execution'
        def runner(name,_expected):
            verify_source_hashes(config)
            verify_calibration_hashes(config)
            weights=dict(config['arms'][name]);c={**config,'normalization':asdict(norm),'loss_weights':weights,'arm':name}
            prov={**provenance,'arm':name,'loss_weights':weights}
            append_coverage_evidence(config,f'BALANCE-{name.upper()}-PRE','Arm preregistered',{'arm':name,'weights':weights,'config_hash':canonical_hash(c),'expected_initial_hashes':expected,
                'record_ids':list(fitting.record_ids),'anchor_ids':list(anchor.record_ids),'updates':2000,'wall_seconds':1800,
                'acceptance':'Original raw energy AND force gates on fitting/anchor; force-only force success reported separately','limits':'n=1; anchor fully fitted; fixed whole-tree initialization calibration; force-only energy gauge unconstrained'})
            factory=None if name=='replay' else partial(build_weighted_step,weights=weights)
            arm=run_coverage_arm(c,fitting,anchor,root/name,prov,step_factory=factory)
            append_coverage_evidence(config,f'BALANCE-{name.upper()}-RESULT','Arm outcome',{
                **{k:arm.get(k) for k in ('status','execution_valid','executed','updates','stop_reason','failure_reason','fit_gate','anchor_gate','elapsed_seconds','frame_exposures','warmup_frame_exposures','initial_state_hashes')},
                'artifact_sha256':{p.name:sha256(p) for p in (root/name).iterdir() if p.is_file()}})
            verify_source_hashes(config)
            verify_calibration_hashes(config)
            return arm
        result=execute_balance_matrix(root,runner,lambda arm:verify_balance_replay(arm,source,config['replay_tolerance']))
        for name,arm in result['arms'].items():
            if not arm.get('executed') or arm.get('controller_exception'):
                append_coverage_evidence(config,f'BALANCE-{name.upper()}-CONTROLLER','Blocked/interrupted arm retained',arm)
        if result.get('replay',{}).get('status')=='failed':append_coverage_evidence(config,'BALANCE-REPLAY-COMPARISON-FAILED','Completed replay comparison failed',result['replay'])
        verify_source_hashes(config)
        verify_calibration_hashes(config)
        result['source_hashes_unchanged']=True
    except Exception as exc:
        result={'status':'failed','execution_valid':False,'balanced_joint_accepted':False,'full_pilot':'pending','failure_reason':f'{type(exc).__name__}: {exc}',
            'stage':stage,'arms':{name:_retain_or_block(root,name,f'{stage}_failure',exc) for name in ARMS},'next_step':'resolve_execution_failure'}
    write_json(root/'summary.json',result)
    append_coverage_evidence(config,'BALANCE-EXECUTION-RESULT','Fixed loss matrix outcome',{
        **{k:result.get(k) for k in ('status','execution_valid','balanced_joint_accepted','force_only_force_accepted','next_step','failure_reason','source_hashes_unchanged')},
        'arm_statuses':{k:v['status'] for k,v in result['arms'].items()},'summary_sha256':sha256(root/'summary.json')})
    return result
