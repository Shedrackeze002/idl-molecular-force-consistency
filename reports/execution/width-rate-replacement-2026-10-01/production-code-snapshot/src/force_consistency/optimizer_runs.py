"""Strict source gates and once-only controller for frozen Adam diagnostics."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from .coverage import take_subset
from .coverage_runs import _arrays,_state_hashes,_predict,_group_score,gate_result,MSE_KEYS,WallBudget
from .data.prepare import array_identity,write_json
from .data.splits import canonical_hash
from .diagnostics import sha256
from .gradient_runs import verify_source_hashes,_evidence
from .gradient_diagnostics import _batches
from .loss_balance_runs import balance_preflight,verify_calibration_hashes
from .models.energy import EnergyModel
from .models.backbone import ModelConfig
from .optimizer_diagnostics import analyze_optimizer_snapshot
from .runs import load_checkpoint,assert_finite,code_provenance
from .train import initialize_state
from .metrics import partition_batch

CASES=('replay','balanced','force-only')
FILES={'config.json','provenance.json','summary.json','subsets.json'}|{f'{name}/{file}' for name in CASES for file in ('config.json','provenance.json','summary.json','history.json','fitting-inputs.npz','anchor-inputs.npz','reference.npz','last.msgpack','checkpoint-verification.json')}
FROZEN={'adam':{'b1':.9,'b2':.999,'eps':1e-8,'eps_root':0.,'nesterov':False},'cases':list(CASES),
        'directions':['stateful_adam','reset_adam_matched_norm','steepest_matched_norm'],'probe_scales':[.1,1.],
        'fd_steps':[1e-4,1e-5],'fd_atol':1e-8,'fd_rtol':1e-4,'gradient_atol':1e-10,'gradient_rtol':1e-8,
        'direction_zero_tolerance':0.,'diagnostic_wall_seconds':600,'replay_tolerance':{'atol':1e-10,'rtol':1e-8}}


def _read(path):return json.loads(Path(path).read_text())


def verify_optimizer_sources(config):
    for key,wanted in FROZEN.items():
        if config.get(key)!=wanted:raise ValueError(f'Frozen optimizer diagnostic setting mismatch: {key}')
    verify_source_hashes(config);calibration=verify_calibration_hashes(config)
    if set(config['loss_balance_sha256'])!=FILES:raise ValueError('Incomplete loss-source allowlist')
    root=Path(config['loss_balance_run'])
    for file,digest in config['loss_balance_sha256'].items():
        if sha256(root/file)!=digest:raise ValueError(f'Loss-source identity mismatch: {file}')
    if sha256(config['loss_balance_compact_path'])!=config['loss_balance_compact_sha256']:
        raise ValueError('Loss compact identity mismatch')
    compact=_read(config['loss_balance_compact_path'])
    if compact['artifacts_sha256']!=config['loss_balance_sha256']:raise ValueError('Loss compact raw index mismatch')
    return compact,calibration


def _metadata(config,compact,calibration):
    root=Path(config['loss_balance_run']);c=_read(root/'config.json');p=_read(root/'provenance.json');summary=_read(root/'summary.json')
    for key,value in c.items():
        if key!='run_directory' and config.get(key)!=value:raise ValueError(f'Loss-source root setting mismatch: {key}')
    if not summary['execution_valid'] or summary['status']!='completed' or not summary['source_hashes_unchanged'] or compact['config']!=c or not compact['execution_valid']:
        raise ValueError('Invalid loss source root/compact')
    if compact['execution_provenance']!={k:v for k,v in p.items() if k!='historical_reference'} or compact['source_summary_sha256']!=sha256(root/'summary.json'):
        raise ValueError('Loss compact provenance mismatch')
    if p['calibration_raw_sources']!=calibration['sources'] or p['calibration_identity']!=config['calibration_sha256']:
        raise ValueError('Loss calibration metadata mismatch')
    expected=p['expected_initial_hashes'];arms={}
    for name in CASES:
        path=root/name;ac=_read(path/'config.json');ap=_read(path/'provenance.json');s=_read(path/'summary.json');v=_read(path/'checkpoint-verification.json')
        if ac!={**c,'normalization':p['normalization'],'loss_weights':config['arms'][name],'arm':name}:
            raise ValueError('Loss arm configuration mismatch')
        if ap.get('config_hash')!=canonical_hash(ac) or ap.get('loss_weights')!=config['arms'][name] or ap.get('arm')!=name:
            raise ValueError('Loss arm provenance/weights mismatch')
        for key,value in p.items():
            if ap.get(key)!=value:raise ValueError(f'Loss arm inherited provenance mismatch: {key}')
        if s['provenance']!=ap or s!=summary['arms'][name] or {k:v for k,v in s.items() if k!='provenance'}!=compact['arms'][name]:raise ValueError('Loss raw/compact arm mismatch')
        if not s['execution_valid'] or not s['executed'] or not s['checkpoint_new_process_verified'] or s['status']!='completed' or s['stop_reason']!='max_updates':
            raise ValueError('Invalid completed loss arm')
        if (s['updates'],s['frame_exposures'],s['warmup_frame_exposures'],s['fitting_frame_count'],s['evaluation_passes'],s['microbatches_per_update'])!=(2000,256000,128,128,22,8):
            raise ValueError('Loss source counters mismatch')
        history=_read(path/'history.json')
        if history!=s['history'] or [x['update'] for x in history]!=sorted([0,250,*range(100,2001,100)]):
            raise ValueError('Loss source history schedule mismatch')
        if history[0]['metrics']!=s['initial_metrics'] or history[-1]['metrics']!=s['final_metrics'] or s['initial_state_hashes']!=expected:
            raise ValueError('Loss source metrics/fresh hash mismatch')
        if v['status']!='passed' or v['update']!=2000 or v['fitting_record_count']!=128 or v['anchor_record_count']!=16:
            raise ValueError('Loss source checkpoint verification mismatch')
        for role,label in [('fitting','fit_gate'),('anchor','anchor_gate')]:
            if s[label]!=gate_result(s['final_metrics'][role],s['initial_metrics'][role],.1):raise ValueError('Loss source immutable gate mismatch')
        arms[name]={'config':ac,'provenance':ap,'summary':s,'verification':v,'initial_metrics':s['initial_metrics']}
    return c,p,arms


def optimizer_preflight(config,check_budget=lambda:None,save_partial=lambda d:None):
    record={'status':'running','compute':{'operations':[],'completed_frame_evaluations':{}},'raw_source_count':100,'compact_source_count':2}
    def keep():save_partial(record)
    def operation(category,frames,fn,numeric=True):
        check_budget();op={'category':category,'planned_frames':frames,'completed_frames':None,'status':'started'}
        record['compute']['operations'].append(op);keep();value=fn()
        if numeric:jax.block_until_ready(value)
        op.update(status='completed',completed_frames=frames)
        counts=record['compute']['completed_frame_evaluations'];counts[category]=counts.get(category,0)+frames
        keep()
        if numeric:assert_finite(value)
        check_budget();return value
    check_budget();compact,calibration=verify_optimizer_sources(config);sourceconfig,p,arms=_metadata(config,compact,calibration);check_budget()
    # All new control metadata gates above precede the existing prepared-array loader.
    train,manifest,norm,selected,historical,cal=operation('parent_source_predictions',208,lambda:balance_preflight(config),False)
    if p['normalization']!=asdict(norm) or p['dataset_identity']!=manifest['identity'] or p['split_hash']!=manifest['split_hash'] or p['precision']!='float64' or p['backend']!='cpu':
        raise ValueError('Loss source dataset/normalization mismatch')
    if _read(Path(config['loss_balance_run'])/'subsets.json')!=selected or p['frozen_subsets_identity']!=config['expected_subsets_identity']:
        raise ValueError('Loss source subsets mismatch')
    fitting=take_subset(train,selected['subsets']['diverse128']['record_ids']);anchor=take_subset(train,selected['subsets']['narrow16']['record_ids'])
    if not set(anchor.record_ids)<=set(fitting.record_ids):raise ValueError('Anchor overlap mismatch')
    model=EnergyModel(ModelConfig(**config['model']))
    check_budget();fresh=initialize_state(model,partition_batch(anchor,np.arange(config['microbatch_size'])),norm,config['seed'],config['learning_rate']);jax.block_until_ready(fresh);check_budget()
    if _state_hashes(fresh)!=p['expected_initial_hashes']:raise ValueError('Loss source fresh state mismatch')
    original=historical['arms']['diverse128']['summary'];states={}
    for name in CASES:
        check_budget();path=Path(config['loss_balance_run'])/name;arm=arms[name];ap=arm['provenance'];s=arm['summary']
        if s['initial_metrics']!=original['initial_metrics']:raise ValueError('Original raw metric baseline mismatch')
        for role,part in [('fitting',fitting),('anchor',anchor)]:
            expected={'arrays':{k:array_identity(a) for k,a in _arrays(part).items()},'record_ids':list(part.record_ids),'memberships':part.memberships}
            if ap['inputs'][role]!=expected:raise ValueError('Loss source input metadata mismatch')
            with np.load(path/f'{role}-inputs.npz',allow_pickle=False) as saved:
                if set(saved.files)!=set(_arrays(part)) or any(array_identity(saved[k])!=expected['arrays'][k] for k in saved.files):raise ValueError('Loss source input arrays mismatch')
        state=load_checkpoint(path/'last.msgpack',expected_provenance=ap,template=fresh)
        if int(state['update'])!=2000 or int(state['optimizer'][0].count)!=2000 or json.loads(state['normalization_json'])!=asdict(norm):raise ValueError('Loss checkpoint counters/normalization mismatch')
        floats=jax.tree.leaves((state['params'],state['optimizer'][0].mu,state['optimizer'][0].nu))
        if any(np.asarray(x).dtype!=np.dtype('float64') for x in floats) or any(np.any(np.asarray(x)<0) for x in jax.tree.leaves(state['optimizer'][0].nu)):
            raise ValueError('Loss checkpoint dtype/variance mismatch')
        with np.load(path/'reference.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'fitting_energy','fitting_forces','anchor_energy','anchor_forces'}:raise ValueError('Loss reference fields mismatch')
            for role,part in [('fitting',fitting),('anchor',anchor)]:
                es=[];fs=[]
                for batch in _batches(part,config['microbatch_size']):
                    e,f=operation('loss_source_predictions',len(batch['energy']),lambda batch=batch:_predict(model)(state['params'],batch))
                    es.append(np.asarray(e));fs.append(np.asarray(f).reshape(len(batch['energy']),*part.forces.shape[1:]))
                e=np.concatenate(es);f=np.concatenate(fs)
                for label,actual in [('energy',e),('forces',f)]:
                    wanted=saved[f'{role}_{label}'];assert_finite(wanted)
                    if actual.shape!=wanted.shape or not np.allclose(actual,wanted,atol=1e-10,rtol=0):raise ValueError('Loss source prediction mismatch')
                observed=_group_score(part,e,f)
                for key in MSE_KEYS:
                    if not np.isclose(observed[key],s['final_metrics'][role][key],atol=1e-10,rtol=1e-8):raise ValueError('Loss source raw metric mismatch')
        states[name]=state;arm['state_hashes']=_state_hashes(state)
    verify_optimizer_sources(config);check_budget()
    record.update(status='completed',arms=arms,historical_reference=historical,normalization=asdict(norm),
                  sources={'coverage':config['source_sha256'],'calibration':calibration['sources'],'loss_balance':config['loss_balance_sha256'],
                           'compact_calibration':config['calibration_sha256'],'compact_loss_balance':config['loss_balance_compact_sha256']})
    keep();return fitting,anchor,model,norm,states,record


def execute_optimizer_cases(root,states,fitting,anchor,model,norm,source,config,check_budget,*,analyzer=analyze_optimizer_snapshot,verify=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);out={};blocked=None
    verify=verify or (lambda:verify_optimizer_sources(config))
    for number,name in enumerate(CASES,1):
        path=root/name;path.mkdir(exist_ok=False);state=states[name];before=_state_hashes(state)
        if blocked is None:
            try:check_budget();verify()
            except Exception as exc:blocked=f'{type(exc).__name__}: {exc}'
        if blocked:
            result={'status':'blocked','executed':False,'verification_valid':False,'stop_reason':blocked,'training_frame_exposures':0}
        else:
            prospective={'case':name,'source_update':int(state['update']),'state_hashes':before,'weights':config['arms'][name],
                'config_hash':canonical_hash(config),'fitting_ids':list(fitting.record_ids),'anchor_ids':list(anchor.record_ids),
                'directions':config['directions'],'scales':config['probe_scales'],'fd_steps':config['fd_steps'],
                'limits':'Temporary local displacements only; no committed updates, winner selection or learning/pilot clearance'}
            write_json(path/'prospective.json',prospective);_evidence(config,f'OPTSTEP-CASE-{number:02d}-PRE','Case preregistered',prospective)
            start=time.perf_counter()
            try:
                local={**config,'_baselines':source['arms'][name]['initial_metrics'],'_check_budget':check_budget,'_save_partial':lambda d:write_json(path/'partial.json',d)}
                observed=analyzer(state,fitting,anchor,model,norm,config['arms'][name],local)
                write_json(path/'observations.json',observed)
                if not observed['verification_valid']:raise ValueError('Snapshot verification failed')
                verify();check_budget();result={**observed,'status':'completed','executed':True}
            except Exception as exc:
                result={**(_read(path/'partial.json') if (path/'partial.json').exists() else {}),'status':'failed','executed':True,'verification_valid':False,
                        'failure_reason':f'{type(exc).__name__}: {exc}','training_frame_exposures':0}
                blocked=f'{name}: {result["failure_reason"]}'
            result['elapsed_seconds']=time.perf_counter()-start
        result['state_hashes_before']=before;result['state_hashes_after']=_state_hashes(state)
        if before!=result['state_hashes_after']:
            result.update(status='failed',verification_valid=False,failure_reason='Source state changed');blocked=f'{name}: source state changed'
        write_json(path/'summary.json',result);out[name]=result;write_json(root/'cases.json',out)
        _evidence(config,f'OPTSTEP-CASE-{number:02d}-RESULT','Case outcome',{**{k:result.get(k) for k in ('status','executed','verification_valid','counts','compute','failure_reason','stop_reason','state_hashes_before','state_hashes_after')},
            'artifact_sha256':{p.name:sha256(p) for p in path.iterdir() if p.is_file()}})
    return out


def diagnose_optimizer_steps(config):
    start=time.perf_counter();root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',config)
    result={'status':'failed','execution_valid':False,'full_pilot':'pending','learning_accepted':False,'training_frame_exposures':0,'stage':'preflight','cases':{}}
    def check():
        if time.perf_counter()-start>=config['diagnostic_wall_seconds']:raise WallBudget('Whole optimizer diagnostic wall budget reached')
    _evidence(config,'OPTSTEP-EXECUTION-PRE','Frozen optimizer diagnostics preregistered',{'config':config,'config_hash':canonical_hash(config),'limits':'Exactly3savedstates; no research retry or training;600s boundary cap'})
    try:
        fitting,anchor,model,norm,states,source=optimizer_preflight(config,check,lambda d:write_json(root/'preflight-partial.json',d))
        write_json(root/'sources.json',source)
        write_json(root/'provenance.json',{**code_provenance(Path(__file__).resolve().parents[2]),'config_hash':canonical_hash(config),'historical_sources':source,'normalization':asdict(norm),'precision':'float64','backend':'cpu'})
        result['stage']='execution'
        result['cases']=execute_optimizer_cases(root,states,fitting,anchor,model,norm,source,config,check)
        verify_optimizer_sources(config);check();result['source_hashes_unchanged']=True
        result['execution_valid']=all(v['status']=='completed' and v['verification_valid'] for v in result['cases'].values())
        result['status']='completed' if result['execution_valid'] else 'failed'
        result['next_step']='separately_review_followup_from_local_evidence' if result['execution_valid'] else 'resolve_verification_or_operational_failure'
    except Exception as exc:
        result.update(status='failed',execution_valid=False,failure_reason=f'{type(exc).__name__}: {exc}',next_step='resolve_verification_or_operational_failure')
        if not result['cases'] and (root/'cases.json').exists():result['cases']=_read(root/'cases.json')
        for name in CASES:
            if name not in result['cases']:
                path=root/name;path.mkdir(exist_ok=True);case={'status':'blocked','executed':False,'verification_valid':False,'stop_reason':result['failure_reason'],'training_frame_exposures':0}
                write_json(path/'summary.json',case);result['cases'][name]=case
    result['elapsed_seconds']=time.perf_counter()-start;write_json(root/'summary.json',result)
    # Output is within the cap too; retain the full observations if a final write crossed it.
    try:check()
    except WallBudget as exc:
        result.update(status='failed',execution_valid=False,failure_reason=str(exc),next_step='resolve_verification_or_operational_failure');write_json(root/'summary.json',result)
    _evidence(config,'OPTSTEP-EXECUTION-RESULT','Frozen optimizer diagnostic outcome',{**{k:result.get(k) for k in ('status','execution_valid','failure_reason','elapsed_seconds','source_hashes_unchanged','next_step')},
        'case_statuses':{k:v['status'] for k,v in result['cases'].items()},'summary_sha256':sha256(root/'summary.json')})
    return result
