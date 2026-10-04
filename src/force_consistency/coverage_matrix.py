"""Preflight, ordered execution and replay gates for the reviewed coverage matrix."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .coverage import select_coverage_subsets,take_subset,verify_coverage_subsets
from .coverage_runs import MSE_KEYS,run_coverage_arm
from .data.prepare import load_training_partition,write_json
from .data.splits import canonical_hash
from .diagnostics import sha256
from .models.backbone import ModelConfig
from .runs import code_provenance

ARMS=('narrow16','diverse16','diverse128')


def verify_replay(arm,reference_history,max_updates,tolerance):
    if not arm['execution_valid'] or arm['updates']!=max_updates or not arm['checkpoint_new_process_verified']:
        raise ValueError('Replay incomplete or operationally invalid')
    current={r['update']:r['metrics']['fitting'] for r in arm['history']}
    expected={r['update']:r for r in reference_history}
    if len(current)!=len(arm['history']) or len(expected)!=len(reference_history): raise ValueError('Duplicate replay history updates')
    scheduled=set(range(0,max_updates+1,100))|{max_updates}
    if not scheduled<=set(expected) or not set(expected)<=set(current): raise ValueError('Missing replay scheduled metrics')
    error=0.
    for update,row in expected.items():
        for key in MSE_KEYS:
            actual=current[update][key];wanted=row[key]
            if not np.isfinite([actual,wanted]).all() or not np.isclose(actual,wanted,**tolerance):
                raise ValueError(f'Replay mismatch: update {update}, {key}: {actual} versus {wanted}')
            error=max(error,abs(actual-wanted))
    return {'status':'passed','compared_updates':sorted(expected),'maximum_absolute_mse_difference':error,'tolerance':tolerance}


def _blocked(root,name,reason):
    result={'status':'blocked','executed':False,'execution_valid':False,'updates':0,'stop_reason':reason,
            'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    path=Path(root)/name;path.mkdir(parents=True,exist_ok=False);write_json(path/'summary.json',result)
    return result


def _failed_arm(root,name,exc):
    """Record an attempted arm without replacing existing research artifacts."""
    path=Path(root)/name;path.mkdir(parents=True,exist_ok=True)
    previous={};history=[];read_errors=[]
    for filename in ('summary.json','history.json'):
        file=path/filename
        if file.exists():
            try:
                saved=json.loads(file.read_text())
                if filename=='summary.json' and isinstance(saved,dict): previous=saved
                elif filename=='history.json' and isinstance(saved,list): history=saved
            except (OSError,ValueError) as error:
                read_errors.append(f'{filename}: {type(error).__name__}: {error}')
    update=previous.get('updates',0)
    if history and isinstance(history[-1],dict): update=history[-1].get('update',update)
    result={**previous,'status':'failed','executed':True,'execution_valid':False,'updates':update,
            'stop_reason':'uncaught_runner_exception','failure_reason':f'{type(exc).__name__}: {exc}',
            'controller_exception':True,'fit_gate':previous.get('fit_gate',{'accepted':False}),
            'anchor_gate':previous.get('anchor_gate',{'accepted':False})}
    if history: result['history']=history
    if previous: result['prior_summary_status']=previous.get('status')
    if read_errors: result['artifact_read_errors']=read_errors
    # A completed/failed arm summary is immutable evidence. The controller's
    # additional failure gets a separate artifact and is reflected at root.
    write_json(path/'controller-failure.json',result)
    if not (path/'summary.json').exists(): write_json(path/'summary.json',result)
    return result


def _retain_or_block(root,name,reason,exc):
    path=Path(root)/name
    if not path.exists(): return _blocked(root,name,reason)
    summary=path/'summary.json'
    if summary.exists():
        try:
            saved=json.loads(summary.read_text())
            if isinstance(saved,dict): return saved
        except (OSError,ValueError): pass
    return _failed_arm(root,name,exc)


def execute_matrix(root,runner,replay):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    result={'status':'failed','execution_valid':False,'full_pilot':'pending','arms':{}}
    blocked_reason=None;expected=None
    for name in ARMS:
        if blocked_reason:
            result['arms'][name]=_blocked(root,name,blocked_reason);continue
        try:
            arm=runner(name,expected);result['arms'][name]=arm
            if name=='narrow16':
                result['replay']=replay(arm)
                expected=arm['initial_state_hashes']
            if not arm['execution_valid']: blocked_reason=f'{name}_operational_or_budget_failure'
        except Exception as exc:
            blocked_reason=f'{name}_replay_or_execution_failure'
            result['failure_reason']=f'{type(exc).__name__}: {exc}'
            if name not in result['arms']: result['arms'][name]=_failed_arm(root,name,exc)
    result['execution_valid']=blocked_reason is None
    result['status']='completed' if result['execution_valid'] else 'failed'
    c=result['arms']['diverse128']
    result['next_step']='separately_review_protocol_amendment' if result['execution_valid'] and c['fit_gate']['accepted'] and c['anchor_gate']['accepted'] else ('architecture_gradient_investigation' if result['execution_valid'] else 'resolve_execution_failure')
    write_json(root/'summary.json',result)
    return result


def append_coverage_evidence(config,record_id,event,details):
    if config.get('evidence_log'):
        with Path(config['evidence_log']).open('a') as file:
            file.write(f'\n## {record_id} — {event} (2026-09-30)\n\nTraining-only diagnostic; no scientific generalization claim; full pilot remains pending.\n\n```json\n')
            file.write(json.dumps(details,sort_keys=True,indent=2,allow_nan=False)+'\n```\n')


def coverage_preflight(config):
    source=Path(config['reference_run'])
    required={'summary.json','history.json','config.json','provenance.json','checkpoint-verification.json'}
    if set(config['reference_sha256'])!=required: raise ValueError('Incomplete replay reference identities')
    for name,digest in config['reference_sha256'].items():
        if sha256(source/name)!=digest: raise ValueError(f'Replay reference identity mismatch: {name}')
    planned=json.loads(Path(config['subsets_manifest']).read_text())
    if planned['identity']!=config['expected_subsets_identity'] or canonical_hash({k:v for k,v in planned.items() if k!='identity'})!=planned['identity']:
        raise ValueError('Frozen subset identity mismatch')
    anchor_file=Path(planned['anchor_source'])
    if sha256(anchor_file)!=planned['anchor_source_sha256']: raise ValueError('Anchor source identity mismatch')
    anchor_source=json.loads(anchor_file.read_text())
    anchor_ids=anchor_source['record_ids']
    if canonical_hash(anchor_ids)!=planned['anchor_identity'] or anchor_source['identity']!=planned['anchor_identity']: raise ValueError('Anchor record identity mismatch')
    historical=json.loads((source/'config.json').read_text());historic_prov=json.loads((source/'provenance.json').read_text())
    for key in ('seed','learning_rate','expected_prepared_identity','expected_split_hash','gate_fraction','backend'):
        if historical[key]!=config[key]: raise ValueError(f'Replay setting mismatch: {key}')
    if ModelConfig(**historical['model'])!=ModelConfig(**config['model']) or historical['batch_size']!=config['microbatch_size']:
        raise ValueError('Replay model or microbatch setting mismatch')
    if config.get('early_stopping',False) or config.get('resume_checkpoint') or config['backend']!='cpu': raise ValueError('Invalid coverage controls')
    history=json.loads((source/'history.json').read_text());summary=json.loads((source/'summary.json').read_text())
    verification=json.loads((source/'checkpoint-verification.json').read_text())
    if summary['updates']!=config['max_updates'] or verification['update']!=config['max_updates'] or verification['status']!='passed':
        raise ValueError('Reference budget or verification mismatch')
    p,manifest,norm=load_training_partition(config['data_directory'],expected_split_hash=config['expected_split_hash'])
    if manifest['identity']!=config['expected_prepared_identity'] or planned['prepared_identity']!=manifest['identity']:
        raise ValueError('Prepared identity mismatch')
    if historical['normalization']!=asdict(norm) or historic_prov['normalization']!=asdict(norm): raise ValueError('Historical normalization mismatch')
    selected=select_coverage_subsets(p,manifest,anchor_ids,planned['selection_seed']);verify_coverage_subsets(selected,planned)
    if anchor_source['indices']!=selected['subsets']['narrow16']['indices']: raise ValueError('Historical anchor index mismatch')
    return p,manifest,norm,selected,history,historic_prov


def diagnose_coverage(config):
    root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()): raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',config)
    stage='preflight'
    try:
        p,manifest,norm,selected,reference,historic_prov=coverage_preflight(config)
        prov={**code_provenance(Path(__file__).resolve().parents[2]),'dataset_identity':manifest['identity'],'split_hash':manifest['split_hash'],
              'archive_sha256':manifest['archive_sha256'],'training_arrays':manifest['partitions']['train']['arrays'],
              'policy_hash':manifest['policy_hash'],'unit_evidence_hash':manifest['unit_evidence_hash'],'normalization':asdict(norm),
              'frozen_subsets_identity':config['expected_subsets_identity'],'config_hash':canonical_hash(config),
              'historical_reference':{'run':config['reference_run'],'hashes':config['reference_sha256'],'provenance':historic_prov},
              'precision':'float64','backend':'cpu','anchor_initial_mse':{k:reference[0][k] for k in MSE_KEYS}}
        write_json(root/'provenance.json',prov);write_json(root/'subsets.json',selected)
        anchor=take_subset(p,selected['subsets']['narrow16']['record_ids'])
        def run(name,expected):
            fitting=take_subset(p,selected['subsets'][name]['record_ids'])
            arm_prov={**prov,'arm':name,'subset_identity':selected['subsets'][name]['identity']}
            if expected is not None: arm_prov['expected_initial_hashes']=expected
            number=ARMS.index(name)+1
            append_coverage_evidence(config,f'COVERAGE-00{number}-PRE',f'{name} trial preregistered',
                                     {'rationale':'Replay' if number==1 else 'Broader connection support' if number==2 else 'Extra within-connection coverage at matched support',
                                      'config':config,'subset':selected['subsets'][name],'run_directory':str(root/name),
                                      'fit_gate':'Both final MSEs < 0.1 own initial','anchor_gate':'Both final anchor MSEs < 0.1 fixed original initial',
                                      'limitations':'n=1; B changes record difficulty; C uses 8x exposures per update; training-only anchor'})
            result=run_coverage_arm({**config,'normalization':asdict(norm)},fitting,anchor,root/name,arm_prov)
            details={k:result.get(k) for k in ('status','execution_valid','updates','stop_reason','failure_reason','fit_gate','anchor_gate','elapsed_seconds','frame_exposures','initial_state_hashes')}
            details['artifact_sha256']={n:sha256(root/name/n) for n in ('summary.json','history.json','last.msgpack','checkpoint-verification.json') if (root/name/n).exists()}
            append_coverage_evidence(config,f'COVERAGE-00{number}-RESULT',f'{name} trial outcome',details)
            return result
        stage='execution'
        result=execute_matrix(root,run,lambda arm:verify_replay(arm,reference,config['max_updates'],config['replay_tolerance']))
        result['provenance']=prov
        for name,arm in result['arms'].items():
            if arm.get('controller_exception'):
                append_coverage_evidence(config,f'COVERAGE-00{ARMS.index(name)+1}-EXCEPTION',f'{name} interrupted attempt retained',arm)
            if not arm.get('executed'):
                append_coverage_evidence(config,f'COVERAGE-00{ARMS.index(name)+1}-BLOCKED',f'{name} not executed',arm)
    except Exception as exc:
        result={'status':'failed','execution_valid':False,'full_pilot':'pending','failure_reason':f'{type(exc).__name__}: {exc}',
                'arms':{name:_retain_or_block(root,name,f'{stage}_failure',exc) for name in ARMS},'next_step':'resolve_execution_failure'}
        append_coverage_evidence(config,f'COVERAGE-{stage.upper()}-FAILED',f'Matrix {stage} failure retained',result)
    write_json(root/'summary.json',result)
    return result
