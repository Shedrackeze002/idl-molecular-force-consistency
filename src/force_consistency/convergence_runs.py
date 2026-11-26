"""Fixed-budget continuation of immutable, verified saved Adam states."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import jax
import numpy as np
from .coverage_runs import (_arrays, _state_hashes, _partition_predictions, coverage_metrics,
                            coverage_probe, gate_result, MSE_KEYS, WallBudget)
from .data.prepare import array_identity
from .data.splits import canonical_hash
from .diagnostics import sha256
from .loss_balance import build_weighted_step
from .optimizer_runs import CASES, optimizer_preflight, verify_optimizer_sources
from .runs import atomic_write, save_checkpoint, load_checkpoint, assert_finite, code_provenance, require_backend
from .metrics import partition_batch
from .models.energy import EnergyModel
from .models.backbone import ModelConfig
from .data.prepare import Normalization
from .train import initialize_state

FROZEN={'start_update':2000,'end_update':6000,'additional_updates':4000,'continuation_wall_seconds':1200,
        'total_wall_seconds':3900,'terminal_window_updates':500,'small_window_relative_change':.01}
OPTFILES={'config.json','provenance.json','sources.json','preflight-partial.json','cases.json','summary.json'}|{
    f'{n}/{f}' for n in CASES for f in ('prospective.json','partial.json','observations.json','summary.json')}


def _read(path):return json.loads(Path(path).read_text())


def atomic_json(path,value):
    atomic_write(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def _evidence(config,identity,event,value):
    if config.get('evidence_log'):
        with Path(config['evidence_log']).open('a') as f:
            f.write(f'\n## {identity} — {event}\n\nFixed continuation controls; fitted anchor, original raw baselines.\n\n```json\n'+json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n```\n')


def _historical_verifier():
    path=Path(__file__).resolve().parents[2]/'scripts/report_optimizer_steps.py'
    spec=importlib.util.spec_from_file_location('optimizer_report_guard',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verified


def verify_convergence_sources(config):
    for key,wanted in FROZEN.items():
        if config.get(key)!=wanted:raise ValueError(f'Frozen continuation setting mismatch: {key}')
    verify_optimizer_sources(config)
    if set(config['optimizer_diagnostic_sha256'])!=OPTFILES:raise ValueError('Incomplete optimizer diagnostic allowlist')
    root=Path(config['optimizer_diagnostic_run'])
    for file,digest in config['optimizer_diagnostic_sha256'].items():
        if sha256(root/file)!=digest:raise ValueError(f'Optimizer diagnostic identity mismatch: {file}')
    if sha256(config['optimizer_diagnostic_compact_path'])!=config['optimizer_diagnostic_compact_sha256']:raise ValueError('Optimizer compact identity mismatch')
    c=_read(root/'config.json');s=_read(root/'summary.json');compact=_read(config['optimizer_diagnostic_compact_path'])
    for key,value in c.items():
        if key!='run_directory' and config.get(key)!=value:raise ValueError(f'Optimizer source setting mismatch: {key}')
    if (not s.get('execution_valid') or s.get('status')!='completed' or not s.get('source_hashes_unchanged')
        or not compact.get('interpretation_valid') or not compact.get('execution_valid') or compact.get('config')!=c
        or compact.get('artifacts_sha256')!=config['optimizer_diagnostic_sha256']
        or compact.get('source_summary_sha256')!=sha256(root/'summary.json')
        or compact.get('cases')!=s.get('cases') or _read(root/'cases.json')!=s.get('cases')
        or set(s.get('cases',{}))!=set(CASES) or compact.get('actual_counts')!={'fd_cells':36,'fd_steps':72,'probes':18}):
        raise ValueError('Optimizer control root/compact metadata mismatch')
    p=_read(root/'provenance.json');sources=_read(root/'sources.json')
    if p.get('config_hash')!=canonical_hash(c) or p.get('historical_sources')!=sources or compact.get('source_verification')!=sources or _read(root/'preflight-partial.json')!=sources:
        raise ValueError('Optimizer source provenance mismatch')
    verified=_historical_verifier();records={}
    for name in CASES:
        case=_read(root/name/'summary.json');observed=_read(root/name/'observations.json');prospective=_read(root/name/'prospective.json')
        if (case!=s['cases'][name] or {k:v for k,v in case.items() if k not in ('executed','elapsed_seconds')}!=observed
            or _read(root/name/'partial.json')!=observed or not verified(case)
            or case.get('counts')!={'fd_cells':12,'fd_steps':24,'probes':6}
            or case.get('update')!=2000 or case.get('adam_count')!=2000 or case.get('weights')!=config['arms'][name]
            or case.get('state_hashes_before')!=sources['arms'][name]['state_hashes']
            or prospective.get('state_hashes')!=case['state_hashes_before'] or prospective.get('config_hash')!=canonical_hash(c)):
            raise ValueError(f'Optimizer case control mismatch: {name}')
        records[name]=case
    return records


def convergence_preflight(config,check_budget=lambda:None,save_partial=lambda d:None):
    check_budget();records=verify_convergence_sources(config);check_budget()
    fitting,anchor,model,norm,states,sources=optimizer_preflight(config,check_budget,save_partial)
    for name in CASES:
        if _state_hashes(states[name])!=records[name]['state_hashes_before']:raise ValueError('Continuation source state mismatch')
    verify_convergence_sources(config);check_budget()
    return fitting,anchor,model,norm,states,sources,records


def schedule(config):
    a,b=config['start_update'],config['end_update'];interval=config['validation_interval']
    return sorted({a,a+1,b,*range((a//interval+1)*interval,b+1,interval)})


def _canonical_state_hashes(state):
    # JAX outputs and Flax template restore can reorder dict keys without changing leaves.
    return _state_hashes({k:jax.tree.map(lambda x:x,state[k]) for k in ('params','optimizer')})


def validate_state(state,norm,update):
    assert_finite(state)
    if int(state['update'])!=update or int(state['optimizer'][0].count)!=update:raise ValueError('Checkpoint update/Adam count mismatch')
    if json.loads(state['normalization_json'])!=asdict(norm):raise ValueError('Checkpoint normalization mismatch')
    floats=jax.tree.leaves((state['params'],state['optimizer'][0].mu,state['optimizer'][0].nu))
    if any(np.asarray(x).dtype!=np.dtype('float64') for x in floats):raise ValueError('Checkpoint float64 mismatch')
    if any(np.any(np.asarray(x)<0) for x in jax.tree.leaves(state['optimizer'][0].nu)):raise ValueError('Checkpoint negative variance')


def run_convergence_arm(config,state,fitting,anchor,model,norm,original_baselines,oracle,root,provenance):
    start=time.perf_counter();root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);incoming=state;state=copy.deepcopy(state);before=_state_hashes(incoming)
    c={**config,'normalization':asdict(norm)};history=[];times=[];eval_times=[];persisted=None
    prov={**provenance,'config_hash':canonical_hash(c),'normalization':asdict(norm),'original_baselines':original_baselines,
          'inputs':{name:{'arrays':{k:array_identity(v) for k,v in _arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships}
                    for name,p in [('fitting',fitting),('anchor',anchor)]}}
    out={'status':'failed','executed':True,'execution_valid':False,'updates':c['start_update'],'additional_updates':0,
         'attempted_updates':0,'pending_update_unknown':False,'warmup_frame_exposures':0,'history':history,
         'original_baselines':original_baselines,'source_state_hashes_before':before,'oracle_verified':False,
         'checkpoint_new_process_verified':False,'source_hashes_unchanged':False,'provenance':prov}
    def check(reserve=0):
        if time.perf_counter()-start+reserve>=c['continuation_wall_seconds']:raise WallBudget('Continuation wall budget reached')
    def persist():
        nonlocal persisted
        snapshot=root/'snapshots'/f'{int(state["update"])}.msgpack'
        # Atomic/versioned files: failed new writes cannot replace a previous verified snapshot.
        save_checkpoint(snapshot,state,prov);atomic_write(root/'last.msgpack',snapshot.read_bytes());persisted=int(state['update'])
        out['persisted_update']=persisted;out['checkpoint_sha256']=sha256(root/'last.msgpack')
    def evaluate():
        t=time.perf_counter();metrics=coverage_metrics(state['params'],fitting,anchor,model,c['microbatch_size']);eval_times.append(time.perf_counter()-t)
        gates={r:gate_result(metrics[r],original_baselines[r],c['gate_fraction']) for r in ('fitting','anchor')}
        row={'update':int(state['update']),'additional_updates':int(state['update'])-c['start_update'],
             'additional_frame_exposures':(int(state['update'])-c['start_update'])*len(fitting.record_ids),
             'frame_exposures':int(state['update'])*len(fitting.record_ids),'elapsed_seconds':time.perf_counter()-start,'metrics':metrics,
             'fit_gate':gates['fitting'],'anchor_gate':gates['anchor'],
             'original_baseline_fraction':{r:gates[r]['fraction_remaining'] for r in gates},
             'weighted_loss':{r:c['loss_weights']['energy']*metrics[r][MSE_KEYS[0]]/norm.energy_loss_scale_eV**2+
                                c['loss_weights']['force']*metrics[r][MSE_KEYS[1]]/norm.force_loss_scale_eV_per_A**2 for r in gates}}
        history.append(row);out.update(final_metrics=metrics,fit_gate=gates['fitting'],anchor_gate=gates['anchor'])
        atomic_json(root/'history.json',history);persist()
        print(json.dumps({'arm':c['arm'],'update':state['update'],'fitting_force_mse':metrics['fitting'][MSE_KEYS[1]],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
        return metrics
    try:
        if (c['end_update']-c['start_update']!=c['additional_updates'] or c['additional_updates']<=0
            or c['validation_interval']<=0 or c['microbatch_size']<=0 or c.get('early_stopping') or c.get('resume_checkpoint')):raise ValueError('Invalid continuation settings')
        check();require_backend('cpu');validate_state(state,norm,c['start_update'])
        atomic_json(root/'config.json',c);atomic_json(root/'provenance.json',prov)
        for name,p in [('fitting',fitting),('anchor',anchor)]:np.savez(root/f'{name}-inputs.npz',**_arrays(p))
        initial=evaluate()
        expected=provenance.get('source_final_metrics',initial)
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:
                if not np.isclose(initial[role][key],expected[role][key],atol=1e-10,rtol=1e-8):raise ValueError('Source metric mismatch')
        batches=[partition_batch(fitting,np.arange(i,min(i+c['microbatch_size'],len(fitting.record_ids)))) for i in range(0,len(fitting.record_ids),c['microbatch_size'])]
        if not batches or not anchor.record_ids:raise ValueError('Empty continuation support')
        check(20+max(eval_times));step=build_weighted_step(model,norm,c['learning_rate'],c['loss_weights']);t=time.perf_counter()
        if hasattr(step,'prepare'):step.prepare(state['params'],state['optimizer'],batches)
        out['compilation_seconds']=time.perf_counter()-t;check(20+max(eval_times));t=time.perf_counter()
        warm=step(state['params'],state['optimizer'],batches);jax.block_until_ready(warm);assert_finite(warm)
        out['warmup_seconds']=time.perf_counter()-t;out['warmup_frame_exposures']=len(fitting.record_ids)
        out['warmup_source_unchanged']=_state_hashes(state)==before
        if not out['warmup_source_unchanged']:raise ValueError('Warmup mutated source')
        for update in range(c['start_update']+1,c['end_update']+1):
            check(20+max(eval_times));out['attempted_updates']+=1;out['pending_update_unknown']=True;t=time.perf_counter()
            pp,oo,mm=step(state['params'],state['optimizer'],batches);jax.block_until_ready((pp,oo,mm));assert_finite((pp,oo,mm))
            candidate={**state,'params':pp,'optimizer':oo,'update':update};validate_state(candidate,norm,update)
            state=candidate;times.append(time.perf_counter()-t);out.update(updates=update,additional_updates=update-c['start_update'],pending_update_unknown=False)
            if update in schedule(c):
                check();metrics=evaluate()
                if update==c['start_update']+1:
                    for role in ('fitting','anchor'):
                        for key in MSE_KEYS:
                            if not np.isclose(metrics[role][key],oracle[role][key],atol=1e-10,rtol=1e-8):raise ValueError('First-step oracle mismatch')
                    out['oracle_verified']=True;out['oracle_metrics']=oracle
                    atomic_json(root/'oracle-verification.json',{'status':'passed','update':update,'metrics':oracle,'atol':1e-10,'rtol':1e-8})
        check();references={}
        for name,p in [('fitting',fitting),('anchor',anchor)]:
            check();e,f=_partition_predictions(state['params'],p,model,c['microbatch_size']);references[f'{name}_energy']=e;references[f'{name}_forces']=f
        np.savez(root/'reference.npz',**references)
        atomic_json(root/'checkpoint-contract.json',{'update':c['end_update'],'adam_count':c['end_update'],'normalization':asdict(norm),
            'state_hashes':_canonical_state_hashes(state),'checkpoint_sha256':sha256(root/'last.msgpack')})
        check();subprocess.run([sys.executable,'-m','force_consistency.convergence_runs','--probe',str(root.resolve())],check=True,capture_output=True,
                              timeout=max(.001,c['continuation_wall_seconds']-(time.perf_counter()-start)))
        receipt=_read(root/'checkpoint-verification.json');check()
        if (receipt.get('status')!='passed' or receipt.get('adam_count')!=c['end_update'] or receipt.get('checkpoint_sha256')!=sha256(root/'last.msgpack')
            or receipt.get('state_hashes')!=_canonical_state_hashes(state)):raise ValueError('Fresh-process receipt mismatch')
        out.update(status='completed',execution_valid=True,stop_reason='fixed_budget',checkpoint_new_process_verified=True,checkpoint_verification=receipt)
    except (KeyboardInterrupt,SystemExit,Exception) as exc:
        out.update(status='incomplete' if isinstance(exc,(WallBudget,subprocess.TimeoutExpired)) else 'failed',execution_valid=False,
                   stop_reason='wall_budget' if isinstance(exc,(WallBudget,subprocess.TimeoutExpired)) else 'failure',failure_reason=f'{type(exc).__name__}: {exc}')
        if isinstance(exc,subprocess.CalledProcessError):out['probe_stderr']=exc.stderr.decode()[-3000:]
        try:
            validate_state(state,norm,int(state['update']));persist()
        except (KeyboardInterrupt,SystemExit,Exception) as final:out['final_persistence_failure']=f'{type(final).__name__}: {final}'
    out['source_state_hashes_after']=_state_hashes(incoming)
    if out['source_state_hashes_after']!=before:out.update(status='failed',execution_valid=False,failure_reason='Source state changed')
    out.update(additional_frame_exposures=out['additional_updates']*len(fitting.record_ids),frame_exposures=out['updates']*len(fitting.record_ids),
               fitting_frame_count=len(fitting.record_ids),anchor_frame_count=len(anchor.record_ids),evaluation_passes=len(eval_times),
               evaluation_frame_count=len(eval_times)*(len(fitting.record_ids)+len(anchor.record_ids)),steady_update_seconds_total=sum(times),
               steady_update_seconds_mean=float(np.mean(times)) if times else None,evaluation_seconds_total=sum(eval_times),
               final_state_hashes=_canonical_state_hashes(state),persisted_update=persisted,elapsed_seconds=time.perf_counter()-start)
    if out['execution_valid'] and out['elapsed_seconds']>=c['continuation_wall_seconds']:out.update(status='failed',execution_valid=False,failure_reason='Output exceeded wall budget')
    try:
        atomic_json(root/'summary.json',out);check()
    except (KeyboardInterrupt,SystemExit,Exception) as exc:
        out.update(status='failed',execution_valid=False,final_persistence_failure=f'{type(exc).__name__}: {exc}')
        try:atomic_json(root/'summary.json',out)
        except (KeyboardInterrupt,SystemExit,Exception):pass
    return out


def convergence_probe(root):
    """Saved-input prediction plus state/count/normalization verification, same subprocess."""
    root=Path(root);c=_read(root/'config.json');prov=_read(root/'provenance.json');contract=_read(root/'checkpoint-contract.json')
    coverage_probe(root)
    with np.load(root/'fitting-inputs.npz',allow_pickle=False) as f:
        from .models.energy import make_batch
        b=make_batch(f['positions'][:c['microbatch_size']],f['atomic_numbers'][:c['microbatch_size']],f['energy_residual'][:c['microbatch_size']],f['forces'][:c['microbatch_size']])
    norm=Normalization(**c['normalization']);model=EnergyModel(ModelConfig(**c['model']))
    template=initialize_state(model,b,norm,c['seed'],c['learning_rate']);state=load_checkpoint(root/'last.msgpack',prov,template=template)
    validate_state(state,norm,c['end_update'])
    if (contract['update']!=c['end_update'] or contract['adam_count']!=c['end_update'] or contract['normalization']!=asdict(norm)
        or contract['state_hashes']!=_canonical_state_hashes(state) or contract['checkpoint_sha256']!=sha256(root/'last.msgpack')):raise ValueError('Checkpoint contract identity mismatch')
    receipt=_read(root/'checkpoint-verification.json')
    atomic_json(root/'checkpoint-verification.json',{**receipt,'adam_count':int(state['optimizer'][0].count),
        'normalization':asdict(norm),'state_hashes':_canonical_state_hashes(state),'checkpoint_sha256':sha256(root/'last.msgpack')})


def execute_convergence_matrix(root,runner,verify=lambda:None,check_budget=lambda:None,config=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);out={};blocked=None;config=config or {}
    for number,name in enumerate(CASES,1):
        path=root/name
        if blocked:
            path.mkdir(exist_ok=True);result={'status':'blocked','executed':False,'execution_valid':False,'stop_reason':blocked,'additional_updates':0,'additional_frame_exposures':0}
        else:
            try:
                check_budget();verify();result=runner(name,path);verify();check_budget();result['source_hashes_unchanged']=True
            except (KeyboardInterrupt,SystemExit,Exception) as exc:
                path.mkdir(exist_ok=True)
                result={**(_read(path/'summary.json') if (path/'summary.json').exists() else {}),'status':'failed','executed':True,'execution_valid':False,
                        'failure_reason':f'{type(exc).__name__}: {exc}'}
                atomic_json(path/'controller-failure.json',result)
            if not result.get('execution_valid'):blocked=f'{name}: {result.get("failure_reason",result.get("stop_reason","invalid execution"))}'
        out[name]=result
        try:
            atomic_json(path/'summary.json',result);atomic_json(root/'arms.json',out)
        except (KeyboardInterrupt,SystemExit,Exception) as exc:
            result.update(status='failed',execution_valid=False,final_persistence_failure=f'{type(exc).__name__}: {exc}')
            blocked=f'{name}: controller persistence failure'
            # Independent receipt survives an arm-summary-specific write failure.
            try:atomic_json(path/'controller-failure.json',result)
            except (KeyboardInterrupt,SystemExit,Exception):pass
            try:atomic_json(root/'arms.json',out)
            except (KeyboardInterrupt,SystemExit,Exception):pass
        _evidence(config,f'CONV-ARM-{number:02d}-RESULT','Immediate continuation outcome',{**{k:result.get(k) for k in ('status','executed','execution_valid','updates','additional_updates','additional_frame_exposures','frame_exposures','failure_reason','stop_reason','oracle_verified','checkpoint_new_process_verified','source_hashes_unchanged','pending_update_unknown','persisted_update')},'artifact_sha256':{str(p.relative_to(path)):sha256(p) for p in path.rglob('*') if p.is_file()}})
    return out


def diagnose_convergence(config):
    start=time.perf_counter();root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);atomic_json(root/'config.json',config)
    result={'status':'failed','execution_valid':False,'learning_accepted':False,'full_pilot':'pending','arms':{},'stage':'preflight'}
    def check():
        if time.perf_counter()-start>=config['total_wall_seconds']:raise WallBudget('Whole continuation wall budget reached')
    _evidence(config,'CONV-EXECUTION-PRE','Fixed continuation preregistered',{'config':config,'config_hash':canonical_hash(config),'limits':'One exact invocation; no adaptive extension or numerical retry; unchanged training-only128/16 controls'})
    try:
        fitting,anchor,model,norm,states,sources,records=convergence_preflight(config,check,lambda d:atomic_json(root/'preflight-partial.json',d))
        atomic_json(root/'sources.json',sources);atomic_json(root/'optimizer-records.json',records)
        provenance={**code_provenance(Path(__file__).resolve().parents[2]),'config_hash':canonical_hash(config),'precision':'float64','backend':'cpu','normalization':asdict(norm),
                    'source_raw_count':118,'source_compact_count':3}
        atomic_json(root/'provenance.json',provenance);original_hashes={n:_state_hashes(s) for n,s in states.items()}
        def verify():
            verify_convergence_sources(config)
            if any(_state_hashes(states[n])!=original_hashes[n] for n in CASES):raise ValueError('Original source states changed')
        def runner(name,path):
            check();arm=sources['arms'][name];oracle=next(p['metrics'] for p in records[name]['probes'] if p['direction']=='stateful_adam' and p['scale']==1.)
            ac={**config,'normalization':asdict(norm),'arm':name,'loss_weights':config['arms'][name]}
            ap={**provenance,'source_state_hashes':original_hashes[name],'source_checkpoint_sha256':config['loss_balance_sha256'][f'{name}/last.msgpack'],
                'source_final_metrics':arm['summary']['final_metrics']}
            prospective={'arm':name,'config_hash':canonical_hash(ac),'source_state_hashes':original_hashes[name],'weights':ac['loss_weights'],
                         'original_baselines':arm['initial_metrics'],'schedule':schedule(ac),'source_checkpoint_sha256':ap['source_checkpoint_sha256']}
            # Separate prospective directory; arm runner owns its protected output directory.
            atomic_json(root/f'{name}-prospective.json',prospective);_evidence(config,f'CONV-ARM-{CASES.index(name)+1:02d}-PRE','Arm preregistered',prospective)
            return run_convergence_arm(ac,states[name],fitting,anchor,model,norm,arm['initial_metrics'],oracle,path,ap)
        result['stage']='execution';result['arms']=execute_convergence_matrix(root,runner,verify,check,config)
        verify();check();result['source_hashes_unchanged']=True
        result['execution_valid']=all(a.get('execution_valid') and a.get('status')=='completed' for a in result['arms'].values())
        result['status']='completed' if result['execution_valid'] else 'failed'
        if result['execution_valid']:
            result['joint_accepted']={n:a['fit_gate']['accepted'] and a['anchor_gate']['accepted'] for n,a in result['arms'].items()}
            result['force_only_force_accepted']=all(result['arms']['force-only']['final_metrics'][r][MSE_KEYS[1]]/sources['arms']['force-only']['initial_metrics'][r][MSE_KEYS[1]]<config['gate_fraction'] for r in ('fitting','anchor'))
            result['learning_accepted']=any(result['joint_accepted'][n] for n in ('replay','balanced'))
    except (KeyboardInterrupt,SystemExit,Exception) as exc:
        result.update(status='failed',execution_valid=False,failure_reason=f'{type(exc).__name__}: {exc}')
        if not result['arms'] and (root/'arms.json').exists():result['arms']=_read(root/'arms.json')
        for n in CASES:
            if n not in result['arms']:
                a={'status':'blocked','executed':False,'execution_valid':False,'additional_updates':0,'additional_frame_exposures':0,'stop_reason':result['failure_reason']}
                atomic_json(root/n/'summary.json',a);result['arms'][n]=a
    result['next_step']='separately_review_budget_evidence' if result['execution_valid'] else 'resolve_retained_operational_failure'
    result['elapsed_seconds']=time.perf_counter()-start;atomic_json(root/'summary.json',result)
    try:check()
    except WallBudget as exc:result.update(status='failed',execution_valid=False,failure_reason=str(exc));atomic_json(root/'summary.json',result)
    _evidence(config,'CONV-EXECUTION-RESULT','Fixed continuation outcome',{**{k:result.get(k) for k in ('status','execution_valid','learning_accepted','joint_accepted','force_only_force_accepted','full_pilot','source_hashes_unchanged','failure_reason','elapsed_seconds')},'summary_sha256':sha256(root/'summary.json')})
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',required=True);convergence_probe(parser.parse_args().probe)
