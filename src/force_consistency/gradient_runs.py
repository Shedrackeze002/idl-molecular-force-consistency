"""Hash-gated train-only controller for one frozen representation/gradient analysis."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from .coverage import select_coverage_subsets,verify_coverage_subsets,take_subset
from .coverage_runs import _state_hashes,_arrays,_partition_predictions,WallBudget
from .data.prepare import load_training_partition,array_identity,write_json
from .data.splits import canonical_hash
from .diagnostics import sha256
from .gradient_diagnostics import analyze_snapshot
from .metrics import partition_batch
from .models.energy import EnergyModel
from .models.backbone import ModelConfig
from .representation import partition_constraints
from .runs import load_checkpoint,assert_finite,require_backend,code_provenance
from .train import initialize_state

ARMS=('narrow16','diverse16','diverse128')
CASE_NAMES=('fresh-A-fitting','fresh-B-fitting','fresh-C-fitting','final-A-fitting',
            'final-B-fitting','final-C-fitting','final-B-anchor','final-C-anchor')
REQUIRED_SOURCES={'config.json','subsets.json','provenance.json','summary.json'}|{
    f'{arm}/{file}' for arm in ARMS for file in ('config.json','provenance.json','summary.json','history.json',
        'fitting-inputs.npz','anchor-inputs.npz','reference.npz','last.msgpack','checkpoint-verification.json')}


def verify_source_hashes(config):
    if set(config['source_sha256'])!=REQUIRED_SOURCES:raise ValueError('Incomplete source hash allowlist')
    for filename,digest in config['source_sha256'].items():
        if sha256(Path(config['source_run'])/filename)!=digest:raise ValueError(f'Source identity mismatch: {filename}')


def _read(root,name):return json.loads((root/name).read_text())


def load_gradient_sources(config,check_budget=lambda:None):
    check_budget();verify_source_hashes(config);check_budget();root=Path(config['source_run'])
    historical=_read(root,'config.json');prov=_read(root,'provenance.json')
    for key in ('seed','learning_rate','microbatch_size','backend','expected_prepared_identity','expected_split_hash','expected_subsets_identity'):
        if config[key]!=historical[key]:raise ValueError(f'Source setting mismatch: {key}')
    if ModelConfig(**config['model'])!=ModelConfig(**historical['model']):raise ValueError('Source model mismatch')
    if config['backend']!='cpu' or config.get('resume_checkpoint') or config.get('early_stopping'):
        raise ValueError('Invalid diagnostic controls')
    if prov['config_hash']!=canonical_hash(historical) or prov['precision']!='float64' or prov['backend']!='cpu':
        raise ValueError('Historical provenance mismatch')
    if not _read(root,'summary.json')['execution_valid']:raise ValueError('Invalid source execution')
    planned=json.loads(Path(config['subsets_manifest']).read_text())
    if planned['identity']!=config['expected_subsets_identity'] or canonical_hash({k:v for k,v in planned.items() if k!='identity'})!=planned['identity']:
        raise ValueError('Subset manifest identity mismatch')
    # The complete hash and setting gate above is deliberately before this loader.
    train,manifest,norm=load_training_partition(config['data_directory'],expected_split_hash=config['expected_split_hash'])
    if manifest['identity']!=config['expected_prepared_identity'] or planned['prepared_identity']!=manifest['identity']:
        raise ValueError('Prepared identity mismatch')
    if prov['normalization']!=asdict(norm):raise ValueError('Source normalization mismatch')
    selected=select_coverage_subsets(train,manifest,planned['subsets']['narrow16']['record_ids'],planned['selection_seed'])
    verify_coverage_subsets(selected,planned)
    if _read(root,'subsets.json')!=selected:raise ValueError('Source subset payload mismatch')
    model=EnergyModel(ModelConfig(**config['model']));anchor=take_subset(train,selected['subsets']['narrow16']['record_ids'])
    require_backend('cpu');check_budget()
    fresh=initialize_state(model,partition_batch(anchor,np.arange(min(config['microbatch_size'],len(anchor.record_ids)))),norm,config['seed'],config['learning_rate'])
    assert_finite(fresh);hashes=_state_hashes(fresh);final={};arm_records={};maxerror=0.
    for arm in ARMS:
        check_budget();path=root/arm;c=_read(path,'config.json');p=_read(path,'provenance.json');s=_read(path,'summary.json');v=_read(path,'checkpoint-verification.json')
        for key in ('seed','learning_rate','microbatch_size','backend','model'):
            if c[key]!=historical[key]:raise ValueError(f'Arm setting mismatch: {arm}/{key}')
        if c['normalization']!=asdict(norm) or p['normalization']!=asdict(norm) or p['config_hash']!=canonical_hash(c):
            raise ValueError('Arm normalization/config identity mismatch')
        if p['backend']!='cpu' or p['precision']!='float64' or p['dataset_identity']!=manifest['identity'] or p['split_hash']!=manifest['split_hash']:
            raise ValueError('Arm dataset/backend mismatch')
        if s['initial_state_hashes']!=hashes or not s['execution_valid'] or s['updates']!=2000 or v['status']!='passed' or v['update']!=2000:
            raise ValueError('Fresh state/final update/source verification mismatch')
        fitting=take_subset(train,selected['subsets'][arm]['record_ids'])
        for name,part in [('fitting',fitting),('anchor',anchor)]:
            expected={'arrays':{k:array_identity(a) for k,a in _arrays(part).items()},'record_ids':list(part.record_ids),'memberships':part.memberships}
            if p['inputs'][name]!=expected:raise ValueError('Source input metadata mismatch')
            with np.load(path/f'{name}-inputs.npz',allow_pickle=False) as saved:
                if set(saved.files)!=set(_arrays(part)) or any(array_identity(saved[k])!=expected['arrays'][k] for k in saved.files):
                    raise ValueError('Saved source arrays mismatch')
        state=load_checkpoint(path/'last.msgpack',expected_provenance=p,template=fresh)
        if int(state['update'])!=2000 or int(state['optimizer'][0].count)!=2000 or json.loads(state['normalization_json'])!=asdict(norm):
            raise ValueError('Checkpoint counter/normalization mismatch')
        with np.load(path/'reference.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'fitting_energy','fitting_forces','anchor_energy','anchor_forces'}:raise ValueError('Reference fields mismatch')
            for name,part in [('fitting',fitting),('anchor',anchor)]:
                check_budget();energy,forces=_partition_predictions(state['params'],part,model,config['microbatch_size'])
                for label,actual in [('energy',energy),('forces',forces)]:
                    wanted=saved[f'{name}_{label}'];assert_finite(wanted)
                    if wanted.shape!=actual.shape or not np.allclose(actual,wanted,atol=1e-10,rtol=0):raise ValueError('Saved prediction mismatch')
                    maxerror=max(maxerror,float(np.max(abs(actual-wanted))))
        final[arm]=state;arm_records[arm]={'config':c,'provenance':p,'summary':s,'verification':v}
    check_budget()
    return train,manifest,norm,selected,fresh,final,{'root':prov,'arms':arm_records,
        'verification':{'initial_state_hashes':hashes,'source_predictions_max_absolute_difference':maxerror}}


def _evidence(config,identity,event,details):
    if config.get('evidence_log'):
        with Path(config['evidence_log']).open('a') as file:
            file.write(f'\n## {identity} — {event}\n\nFrozen training-only diagnostics; no optimizer update or learning acceptance.\n\n```json\n')
            file.write(json.dumps(details,indent=2,sort_keys=True,allow_nan=False)+'\n```\n')


def execute_gradient_cases(root,cases,model,norm,config,check_budget,analyzer=analyze_snapshot):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);out={};blocked=None
    for number,(name,state,part) in enumerate(cases,1):
        path=root/name;path.mkdir(exist_ok=False);attempted=False;before=_state_hashes(state)
        if blocked is None:
            try:check_budget()
            except Exception as exc:blocked=f'{type(exc).__name__}: {exc}'
        if blocked is not None:
            result={'status':'blocked','executed':False,'verification_valid':False,'stop_reason':blocked,
                    'record_ids':list(part.record_ids),'record_count':len(part.record_ids)}
        else:
            prospective={'case':name,'update':int(state['update']),'record_ids':list(part.record_ids),'record_count':len(part.record_ids),
                         'source_hashes':before,'config_hash':canonical_hash(config),'rationale':'Local split loss gradients at frozen incoming parameters',
                         'acceptance':'Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change'}
            write_json(path/'prospective.json',prospective);_evidence(config,f'GRADIENT-CASE-{number:02d}-PRE','Case preregistered',prospective)
            attempted=True;t=time.perf_counter()
            try:
                local={**config,'_check_budget':check_budget,'_save_partial':lambda d:write_json(path/'partial.json',d)}
                observed=analyzer(state,part,model,norm,local)
                # Retain completed observations before any post-compute budget failure.
                write_json(path/'observations.json',observed)
                if not observed['verification_valid']:raise ValueError('Snapshot verification failed')
                check_budget()
                result={**observed,'status':'completed','executed':True}
            except Exception as exc:
                result={'status':'failed','executed':attempted,'verification_valid':False,'failure_reason':f'{type(exc).__name__}: {exc}',
                        'record_ids':list(part.record_ids),'record_count':len(part.record_ids)}
                blocked=f'{name}: {result["failure_reason"]}'
            result['elapsed_seconds']=time.perf_counter()-t
        result['state_hashes_before']=before;result['state_hashes_after']=_state_hashes(state)
        if result['state_hashes_before']!=result['state_hashes_after']:
            result.update(status='failed',verification_valid=False,failure_reason='Source state changed');blocked=f'{name}: source state changed'
        write_json(path/'summary.json',result);out[name]=result;write_json(root/'cases.json',out)
        _evidence(config,f'GRADIENT-CASE-{number:02d}-RESULT','Case outcome',
            {**{k:result.get(k) for k in ('status','executed','verification_valid','failure_reason','stop_reason','elapsed_seconds','state_hashes_before','state_hashes_after')},
             'summary_sha256':sha256(path/'summary.json'),'retained_artifacts':{p.name:sha256(p) for p in path.iterdir() if p.is_file()}})
    return out


def diagnose_gradients(config):
    start=time.perf_counter();root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',config)
    summary={'status':'failed','execution_valid':False,'full_pilot':'pending','learning_accepted':False,'stage':'preflight','cases':{}}
    phase={'status':'unexecuted','subsets':{},'prediction_checks':{}}
    def check_budget():
        if time.perf_counter()-start+config.get('output_reserve_seconds',10)>=config['wall_seconds']:
            raise WallBudget('Whole diagnostic wall budget reached')
    _evidence(config,'GRADIENT-EXECUTION-PRE','Frozen analysis preregistered',{'config':config,'source_config_hash':canonical_hash(config),'limits':'One bounded execution; no retries or training'})
    try:
        train,manifest,norm,selected,fresh,final,historical=load_gradient_sources(config,check_budget)
        check_budget();model=EnergyModel(ModelConfig(**config['model']))
        prov={**code_provenance(Path(__file__).resolve().parents[2]),'config_hash':canonical_hash(config),'normalization':asdict(norm),
              'dataset_identity':manifest['identity'],'split_hash':manifest['split_hash'],'frozen_subsets_identity':config['expected_subsets_identity'],
              'historical_sources':historical,'source_sha256':config['source_sha256'],'precision':'float64','backend':'cpu'}
        write_json(root/'provenance.json',prov);write_json(root/'subsets.json',selected)
        fitting={k:take_subset(train,selected['subsets'][k]['record_ids']) for k in ARMS};anchor=fitting['narrow16']
        summary['stage']='phase-A';phase['status']='running'
        _evidence(config,'GRADIENT-PHASE-A-PRE','Necessary constraints preregistered',{'config_hash':canonical_hash(config),'subsets':selected,
             'acceptance':'Projection/reconstruction checks, saved prediction leakage<=1e-8 and raw-MSE decomposition1e-10/1e-8; original gates unchanged'})
        for arm in ARMS:
            check_budget();constraints=partition_constraints(fitting[arm]);gate=historical['arms'][arm]['summary']['fit_gate']['threshold_mse']['force_mse_eV2_per_A2']
            constraints['immutable_force_gate_mse']=gate;constraints['floor_obstructs_gate']=constraints['pooled']['floor_mse']>gate+1e-10
            phase['subsets'][arm]=constraints;write_json(root/'phase-A.json',phase)
            for label,part in [('fitting',fitting[arm]),('anchor',anchor)]:
                check_budget();_,predicted=_partition_predictions(final[arm]['params'],part,model,config['microbatch_size'])
                d=partition_constraints(part,predicted);phase['prediction_checks'][f'{arm}-{label}']=d['prediction_checks'];write_json(root/'phase-A.json',phase)
                if not d['prediction_checks']['valid']:raise ValueError('Prediction symmetry/decomposition verification failed')
            for row in constraints['records']:
                if row['checks']['reconstruction_max']>1e-10 or row['checks']['orthogonality_max']>1e-10 or abs(row['checks']['pythagorean_error'])>1e-10:
                    raise ValueError('Projection algebra verification failed')
        phase['status']='completed';write_json(root/'phase-A.json',phase)
        _evidence(config,'GRADIENT-PHASE-A-RESULT','Necessary constraint outcome',{'status':phase['status'],'artifact_sha256':sha256(root/'phase-A.json'),
            'pooled':{k:{**v['pooled'],'gate':v['immutable_force_gate_mse'],'floor_obstructs_gate':v['floor_obstructs_gate']} for k,v in phase['subsets'].items()}})
        summary['stage']='phase-B'
        cases=([(CASE_NAMES[i],fresh,fitting[k]) for i,k in enumerate(ARMS)]+[(CASE_NAMES[i+3],final[k],fitting[k]) for i,k in enumerate(ARMS)]+
            [(CASE_NAMES[6],final['diverse16'],anchor),(CASE_NAMES[7],final['diverse128'],anchor)])
        summary['cases']=execute_gradient_cases(root,cases,model,norm,config,check_budget)
        verify_source_hashes(config);check_budget();summary['source_hashes_unchanged']=True
        summary['execution_valid']=all(v['status']=='completed' for v in summary['cases'].values())
        summary['status']='completed' if summary['execution_valid'] else 'failed'
        summary['next_step']='separately_review_source_model_target_contract' if any(v['floor_obstructs_gate'] for v in phase['subsets'].values()) else 'separately_review_intervention_from_local_evidence'
    except Exception as exc:
        summary['failure_reason']=f'{type(exc).__name__}: {exc}'
        if phase['status']=='running':phase.update(status='failed',failure_reason=summary['failure_reason'])
        if not summary['cases'] and (root/'cases.json').exists():summary['cases']=_read(root,'cases.json')
        for name in CASE_NAMES:
            if name not in summary['cases']:
                path=root/name;path.mkdir(exist_ok=True);result={'status':'blocked','executed':False,'verification_valid':False,'stop_reason':summary['failure_reason']}
                write_json(path/'summary.json',result);summary['cases'][name]=result
        summary['next_step']='resolve_verification_or_operational_failure'
    write_json(root/'phase-A.json',phase);summary['elapsed_seconds']=time.perf_counter()-start
    summary['phase_A_status']=phase['status'];write_json(root/'summary.json',summary)
    _evidence(config,'GRADIENT-EXECUTION-RESULT','Frozen analysis outcome',{'status':summary['status'],'execution_valid':summary['execution_valid'],
        'elapsed_seconds':summary['elapsed_seconds'],'failure_reason':summary.get('failure_reason'),'phase_A_status':phase['status'],
        'case_statuses':{k:v['status'] for k,v in summary['cases'].items()},'summary_sha256':sha256(root/'summary.json'),'next_step':summary['next_step']})
    return summary
