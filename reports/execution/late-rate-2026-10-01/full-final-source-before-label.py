"""Bounded late-rate diagnostic, preserving original moments and raw acceptance."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import copy
import subprocess
import sys

import jax
from flax import serialization

import numpy as np
from .data.prepare import Normalization,PreparedPartition,array_identity
from .data.splits import canonical_hash
from .coverage_runs import _arrays,_partition_predictions,_group_score,coverage_metrics,gate_result
from .loss_balance import build_weighted_step,weighted_loss
from .metrics import partition_batch
from .models.backbone import ModelConfig
from .models.energy import EnergyModel
from .runs import atomic_write,save_checkpoint,load_checkpoint,_read_checkpoint,assert_finite,require_backend,code_provenance
from .train import initialize_state
from .width_rate_runs import canonical_state_hashes as state_hashes,validate_state,independent_adam_step

PARENT='runs/aspirin-width-rate-replacement-2026-10-01/width32-lr1e-3'
PIN_PATH='reports/execution/late-rate-2026-10-01/source-pins.json'
PIN_HASH='55fb10bef1abc25783d8b6376dbfcb4896da71db1f6216210a01249224b42e30'
MSE_KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')
WEIGHTS={'energy':.014584365670390719,'force':1.9854156343296092}
SCORES=sorted({6000,6001,8001,*range(6100,10001,100)})


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(value):return json.loads(json.dumps(value,allow_nan=False))
def atomic_json(path,value):atomic_write(path,(json.dumps(normalized(value),sort_keys=True,indent=2)+'\n').encode())


def rate_for_update(arm,update):
    if arm not in ('constant','late_decay') or not 6001<=update<=10000:
        raise ValueError('Unreviewed arm/update')
    return .001 if arm=='constant' else .0003 if update<=8000 else .0001


def validate_contract(c):
    wanted={'status':'planned_unexecuted','parent_directory':PARENT,'source_pins':PIN_PATH,
        'source_pins_sha256':PIN_HASH,'run_directory':'runs/aspirin-late-rate-2026-10-01',
        'evidence_log':'reports/PAPER_EVIDENCE_LOG.md','start_update':6000,'end_update':10000,
        'score_updates':SCORES,'arms':['constant','late_decay'],'arm_wall_seconds':2700,
        'total_wall_seconds':6000,'finalization_reserve_seconds':120,'gate_fraction':.1,
        'microbatch_size':16,'seed':0,'backend':'cpu','checkpoint_prediction_atol':1e-10,
        'loss_weights':WEIGHTS,'learning_rates':{'constant':[[6001,.001]],
            'late_decay':[[6001,.0003],[8001,.0001]]},'maximum_invocations':1,'early_stopping':False}
    if normalized(c)!=wanted:raise ValueError('Reviewed contract changed')


def verify_pins(pins):
    for name,sha in pins.items():
        p=Path(name)
        if not p.is_file() or p.is_symlink() or digest(p)!=sha:
            raise ValueError(f'Pinned source changed: {name}')


def residual_summary(energy,target,forces,force_target):
    er=np.asarray(energy)-target;fr=np.asarray(forces)-force_target
    if not len(er) or not np.isfinite(er).all() or not np.isfinite(fr).all():
        raise ValueError('Empty or nonfinite residuals')
    return {'energy_mse_eV2':float(np.mean(er**2)),
        'signed_mean_energy_error_eV':float(np.mean(er)),
        'mean_energy_error_squared_eV2':float(np.mean(er)**2),
        'centered_energy_error_variance_eV2':float(np.var(er)),
        'force_mse_eV2_per_A2':float(np.mean(fr**2))}


def saved_observations(parent):
    parent=Path(parent);history=read(parent/'histories/6000.json');prov=read(parent/'provenance.json')
    with np.load(parent/'reference.npz',allow_pickle=False) as r,np.load(parent/'fitting-inputs.npz',allow_pickle=False) as p:
        energy=np.array(r['fitting_energy']);forces=np.array(r['fitting_forces'])
        target=np.array(p['energy_residual']);ftarget=np.array(p['forces'])
    end=residual_summary(energy,target,forces,ftarget)
    for key in MSE_KEYS:
        if not np.isclose(end[key],history[-1]['metrics']['fitting'][key],atol=1e-12,rtol=1e-10):
            raise ValueError('Saved prediction metric mismatch')
    decomposition=end['mean_energy_error_squared_eV2']+end['centered_energy_error_variance_eV2']
    if not np.isclose(decomposition,end[MSE_KEYS[0]],atol=1e-12,rtol=1e-10):raise ValueError('Residual decomposition mismatch')
    per_record=[{'record_id':rid,**residual_summary(energy[i:i+1],target[i:i+1],forces[i:i+1],ftarget[i:i+1])}
                for i,rid in enumerate(prov['inputs']['fitting']['record_ids'])]
    groups=history[-1]['metrics']['fitting']['per_group'];count=sum(g['record_count'] for g in groups.values())
    if count!=len(energy):raise ValueError('Overlapping group reconstruction unsupported')
    error=max(abs(sum(g['record_count']*g[k] for g in groups.values())/count-end[k]) for k in MSE_KEYS)
    if error>1e-12:raise ValueError('Group aggregate mismatch')
    peak=max(history,key=lambda r:r['metrics']['fitting'][MSE_KEYS[0]])
    adjacent=[{'update':r['update'],'fitting':{k:r['metrics']['fitting'][k] for k in MSE_KEYS},
               'per_group':r['metrics']['fitting']['per_group']} for r in history if r['update'] in (1900,2000,2100)]
    return {'status':'saved_only','new_training_updates':0,'new_prediction_frames':0,
        'endpoint':end,'endpoint_per_record':per_record,'endpoint_per_group':groups,
        'per_group_reconstruction_max_error':error,
        'spike':{'sampled_peak_update':peak['update'],'adjacent_observations':adjacent},
        'limits':['Sparse saved observations do not locate every spike or establish its mechanism.',
            'Residual mean/variance is descriptive; raw MSE gates are unchanged.',
            'This analysis does not amend the frozen schedule.']}


def verify_code(code):
    verify_pins(code['code_files'])
    if 'code_fingerprint' in code and canonical_hash(code['code_files'])!=code['code_fingerprint']:
        raise ValueError('Code fingerprint disagrees with recorded files')
    if 'lock_sha256' in code and code['code_files'].get('uv.lock')!=code['lock_sha256']:
        raise ValueError('Recorded lock identity disagrees')
    # Revision identifies launch ancestry; file hashes remain authoritative after documentation commits.
    if 'revision' in code and (len(code['revision'])!=40 or any(x not in '0123456789abcdef' for x in code['revision'])):
        raise ValueError('Invalid full launch revision')
    if 'dependencies' in code:
        from importlib.metadata import version
        if {k:version(k) for k in code['dependencies']}!=code['dependencies']:
            raise ValueError('Execution dependency identity changed')
    if 'python' in code:
        import platform
        if platform.python_version()!=code['python']:raise ValueError('Python identity changed')


def _parts(root,prov):
    parts={}
    for role in ('fitting','anchor'):
        meta=prov['inputs'][role];path=Path(root)/f'{role}-inputs.npz'
        if digest(path)!=prov['input_files_sha256'][role]:raise ValueError('Input-file identity changed')
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!=set(meta['arrays']) or any(array_identity(data[k])!=meta['arrays'][k] for k in data.files):
                raise ValueError('Input array identity changed')
            parts[role]=PreparedPartition(**{k:np.array(data[k]) for k in data.files},
                record_ids=tuple(meta['record_ids']),memberships=meta['memberships'])
    return parts


def _prediction_error(state,parts,model,reference,microbatch,tol):
    maximum=0.
    with np.load(reference,allow_pickle=False) as ref:
        if set(ref.files)!={f'{r}_{k}' for r in ('fitting','anchor') for k in ('energy','forces')}:
            raise ValueError('Reference array names changed')
        for role,p in parts.items():
            e,f=_partition_predictions(state['params'],p,model,microbatch)
            for actual,key in [(e,f'{role}_energy'),(f,f'{role}_forces')]:
                if actual.shape!=ref[key].shape:raise ValueError('Reference shape mismatch')
                assert_finite((actual,ref[key]))
                maximum=max(maximum,float(np.max(np.abs(actual-ref[key]))))
    if maximum>tol:raise ValueError('Checkpoint prediction reproduction failed')
    return maximum


def load_parent(c):
    validate_contract(c);require_backend('cpu')
    if digest(c['source_pins'])!=c['source_pins_sha256']:raise ValueError('Pin map changed')
    pins=read(c['source_pins']);verify_pins(pins);root=Path(c['parent_directory'])
    prov=read(root/'provenance.json');receipt=read(root/'commit.json');contract=read(root/'checkpoint-contract.json')
    summary=read(root/'summary.json');child=read(root/'checkpoint-verification.json');completion=read(root/'probe-completion.json')
    history=read(root/receipt['history']);parent_config=read(root/'config.json');case=read(root/'case.json')
    if (not summary['execution_valid'] or summary['status']!='completed' or receipt!=contract['commit']
        or receipt['update']!=6000 or receipt['checkpoint']!='snapshots/6000.msgpack'
        or receipt['history']!='histories/6000.json' or digest(root/receipt['checkpoint'])!=receipt['checkpoint_sha256']
        or digest(root/receipt['history'])!=receipt['history_sha256']
        or prov['original_baselines']!=history[0]['metrics'] or history[0]['update']!=0 or history[-1]['update']!=6000
        or summary['final_metrics']!=history[-1]['metrics'] or child['status']!='passed'
        or child['state_hashes']!=receipt['state_hashes'] or child['checkpoint_sha256']!=receipt['checkpoint_sha256']
        or child['history_sha256']!=receipt['history_sha256'] or child['adam_count']!=6000 or child['prediction_frames']!=144
        or completion['status']!='passed' or completion['verification_sha256']!=digest(root/'checkpoint-verification.json')
        or case!=contract['case'] or case!=prov['case'] or case['features']!=32 or case['learning_rate']!=.001
        or canonical_hash(parent_config)!=prov['config_identity'] or child['config_identity']!=prov['config_identity']
        or contract['inputs']!=prov['inputs'] or contract['reference_sha256']!=digest(root/'reference.npz')
        or parent_config['loss_weights']!=WEIGHTS or parent_config['seed']!=0 or parent_config['microbatch_size']!=16):
        raise ValueError('Parent ancestry/receipt contract mismatch')
    model_config={'features':32,'iterations':2,'max_degree':1,'radial_basis':8,'cutoff_A':5.,'energy_readout_init':'zero'}
    if prov['model']!=model_config or prov['precision']!='float64' or prov['backend']!='cpu':raise ValueError('Parent model/backend changed')
    parts=_parts(root,prov);fit,anchor=parts['fitting'],parts['anchor']
    if (len(fit.record_ids),len(anchor.record_ids))!=(128,16) or not set(anchor.record_ids)<=set(fit.record_ids):
        raise ValueError('Parent support changed')
    norm=Normalization(**prov['normalization']);model=EnergyModel(ModelConfig(**prov['model']))
    template=initialize_state(model,partition_batch(anchor,np.arange(16)),norm,0,.001)
    state=load_checkpoint(root/receipt['checkpoint'],prov,template=template);validate_state(state,norm,6000)
    if state_hashes(state)!=receipt['state_hashes'] or not any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(state['optimizer'][0].mu)):
        raise ValueError('Parent state/moments changed')
    error=_prediction_error(state,parts,model,root/'reference.npz',16,1e-10)
    metrics=coverage_metrics(state['params'],fit,anchor,model,16)
    for role in ('fitting','anchor'):
        for key in MSE_KEYS:
            if not np.isclose(metrics[role][key],history[-1]['metrics'][role][key],atol=1e-10,rtol=1e-8):
                raise ValueError('Parent raw endpoint metric mismatch')
    verify_pins(pins)
    return {'state':state,'fitting':fit,'anchor':anchor,'model':model,'norm':norm,
        'baselines':prov['original_baselines'],'source_pins':pins,'source_prediction_max_error':error,
        'parent_provenance_sha256':digest(root/'provenance.json'),'parent_checkpoint_sha256':receipt['checkpoint_sha256'],
        'parent_metrics':history[-1]['metrics'],'parent_source_reproduction_frames':144,
        'parent_endpoint_scoring_frames':144,'template_initializer_frames':16,
        'code':normalized(code_provenance(Path.cwd()))}


def _verify_parent_binding(c,prov,initial_row=None):
    """Saved-only ancestry check; no initialization, training or prediction."""
    validate_contract(c)
    if digest(c['source_pins'])!=c['source_pins_sha256']:raise ValueError('Frozen pin manifest changed')
    pins=read(c['source_pins']);verify_pins(pins);root=Path(c['parent_directory'])
    pp=read(root/'provenance.json');receipt=read(root/'commit.json');history=read(root/receipt['history'])
    if (prov['source_pins']!=pins or prov['parent_checkpoint_sha256']!=receipt['checkpoint_sha256']
        or prov['parent_provenance_sha256']!=digest(root/'provenance.json')
        or prov['initial_state_hashes']!=receipt['state_hashes']
        or any(prov[k]!=pp[k] for k in ('model','normalization','inputs','original_baselines'))
        or receipt['update']!=6000 or history[0]['metrics']!=prov['original_baselines']):
        raise ValueError('Saved lineage differs from frozen parent')
    if initial_row is not None:
        if initial_row['update']!=6000 or initial_row['state_hashes']!=receipt['state_hashes']:
            raise ValueError('Initial history is not the frozen parent')
        if initial_row['metrics']!=history[-1]['metrics']:raise ValueError('Initial raw metrics differ from parent')


def _close_trees(left,right):
    a,at=jax.tree.flatten(left);b,bt=jax.tree.flatten(right)
    if at!=bt:raise ValueError('Oracle tree structure mismatch')
    maximum=0.
    for x,y in zip(a,b,strict=True):
        x,y=np.asarray(x),np.asarray(y)
        if x.shape!=y.shape or not np.allclose(x,y,atol=1e-10,rtol=1e-8):raise ValueError('Independent Adam oracle mismatch')
        maximum=max(maximum,float(np.max(np.abs(x-y))))
    return maximum


def _boundary(state,rate,step,batches,full,model,norm):
    validate_state(state,norm,int(state['update']));before=state_hashes(state)
    gradient=jax.jit(jax.grad(lambda p:weighted_loss(p,full,model,norm,WEIGHTS)[0]))(state['params'])
    jax.block_until_ready(gradient);assert_finite(gradient)
    rp,ro=independent_adam_step(state['params'],state['optimizer'],gradient,rate)
    ap,ao,metrics=step(state['params'],state['optimizer'],batches);jax.block_until_ready((ap,ao,metrics));assert_finite((ap,ao,metrics))
    discrepancy=_close_trees((ap,ao),(rp,ro));candidate={**state,'params':ap,'optimizer':ao,'update':int(state['update'])+1}
    validate_state(candidate,norm,candidate['update'])
    if state_hashes(state)!=before:raise ValueError('Boundary probe mutated incoming state')
    return candidate,{'status':'passed','incoming_update':int(state['update']),'outgoing_update':candidate['update'],
        'incoming_state_hashes':before,'proposal_state_hashes':state_hashes(candidate),'rate':rate,
        'maximum_absolute_difference':discrepancy,'atol':1e-10,'rtol':1e-8,'discarded':True,
        'fullbatch_gradient_frames':len(full['energy']),'accumulated_gradient_frames':sum(len(b['energy']) for b in batches),
        'independent_numpy_proposals':1,'accumulated_proposals':1}


def _commit(root,state,history,prov):
    u=int(state['update']);cp=root/'snapshots'/f'{u}.msgpack';hp=root/'histories'/f'{u}.json'
    if cp.exists() or hp.exists():raise FileExistsError('Immutable snapshot version exists')
    save_checkpoint(cp,state,prov);atomic_json(hp,history)
    receipt={'update':u,'checkpoint':str(cp.relative_to(root)),'history':str(hp.relative_to(root)),
        'checkpoint_sha256':digest(cp),'history_sha256':digest(hp),'state_hashes':state_hashes(state)}
    atomic_json(root/'commit.json',receipt)
    return receipt


def _launch_probe(root,timeout):
    subprocess.run([sys.executable,'-m','force_consistency.late_rate','--probe',str(root.resolve())],
        check=True,capture_output=True,timeout=timeout)
    return read(root/'checkpoint-verification.json')


def _run_arm(c,arm,prepared,root,deadline,*,probe_runner=_launch_probe):
    """Internal engine; production entry points validate the frozen contract first."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    arm_deadline=min(deadline,start+c['arm_wall_seconds']);reserve=c['finalization_reserve_seconds']
    state=copy.deepcopy(prepared['state']);original_hashes=state_hashes(state);norm=prepared['norm'];model=prepared['model']
    parts={r:prepared[r] for r in ('fitting','anchor')};count=len(parts['fitting'].record_ids)
    history=[];oracles={};receipt=None;computed=0;attempted=0;pending=False
    out={'status':'failed','execution_valid':False,'arm':arm,'additional_updates':0,
        'persisted_additional_updates':0,'pending_update_unknown':False,'warmup_status':'not_started',
        'warmup_frame_exposures':0,'oracle_status':'not_started','probe_status':'not_started',
        'initial_state_hashes':original_hashes,
        'parent_source_reproduction_frames':prepared.get('parent_source_reproduction_frames',0),
        'parent_endpoint_scoring_frames':prepared.get('parent_endpoint_scoring_frames',0),
        'parent_load_prediction_frames':prepared.get('parent_source_reproduction_frames',0)+prepared.get('parent_endpoint_scoring_frames',0),
        'template_initializer_frames':prepared.get('template_initializer_frames',0),
        'scoring_prediction_frames':0,'scoring_status':'not_started','reference_prediction_frames':0,'reference_status':'not_started',
        'template_initializations':1 if prepared.get('template_initializer_frames') else 0,
        'source_prediction_max_error':prepared.get('source_prediction_max_error')}
    def check(final=False):
        if time.perf_counter()+(0 if final else reserve)>=arm_deadline:raise TimeoutError('Continuation wall budget/reserve exhausted')
    def phase(name,status='started',**details):
        atomic_json(root/'phase.json',{'phase':name,'status':status,'last_persisted_update':receipt['update'] if receipt else None,
            'exact_computed_additional_updates':computed,'attempted_additional_updates':attempted,**details})
    try:
        check();verify_pins(prepared['source_pins']);verify_code(prepared['code']);validate_state(state,norm,c['start_update'])
        if not any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(state['optimizer'][0].mu)):raise ValueError('Continuation requires nonzero moments')
        atomic_json(root/'config.json',c)
        inputs={r:{'arrays':{k:array_identity(v) for k,v in _arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships} for r,p in parts.items()}
        for r,p in parts.items():np.savez(root/f'{r}-inputs.npz',**_arrays(p))
        prov=normalized({'config_identity':canonical_hash(c),'arm':arm,'model':asdict(model.config),'normalization':asdict(norm),
            'inputs':inputs,'input_files_sha256':{r:digest(root/f'{r}-inputs.npz') for r in parts},
            'original_baselines':prepared['baselines'],'initial_state_hashes':original_hashes,
            'parent_provenance_sha256':prepared.get('parent_provenance_sha256'),
            'parent_checkpoint_sha256':prepared.get('parent_checkpoint_sha256'),
            'code':prepared['code'],'source_pins':prepared['source_pins'],'precision':'float64','backend':'cpu'})
        atomic_json(root/'provenance.json',prov)
        def score(rate=None):
            nonlocal receipt
            check(final=True);phase('score');out['scoring_status']='started'
            metrics=coverage_metrics(state['params'],parts['fitting'],parts['anchor'],model,c['microbatch_size'])
            assert_finite(metrics)
            out['scoring_prediction_frames']+=sum(len(p.record_ids) for p in parts.values());out['scoring_status']='completed'
            row={'update':int(state['update']),'additional_updates':int(state['update'])-c['start_update'],
                'metrics':metrics,'state_hashes':state_hashes(state),'rate':rate}
            for r in parts:row[r+'_gate']=gate_result(metrics[r],prepared['baselines'][r],c['gate_fraction'])
            history.append(row);receipt=_commit(root,state,history,prov);phase('score','completed');check(final=True)
            print(json.dumps({'arm':arm,'update':int(state['update']),'energy_fraction':row['fitting_gate']['fraction_remaining'][MSE_KEYS[0]],
                'force_fraction':row['fitting_gate']['fraction_remaining'][MSE_KEYS[1]],'elapsed_seconds':time.perf_counter()-start}),flush=True)
        score()
        full=partition_batch(parts['fitting'],np.arange(count));batches=[partition_batch(parts['fitting'],np.arange(i,min(i+c['microbatch_size'],count))) for i in range(0,count,c['microbatch_size'])]
        steps={}
        def get_step(rate):
            if rate not in steps:
                check();phase('compile',rate=rate);s=build_weighted_step(model,norm,rate,WEIGHTS)
                s.prepare(state['params'],state['optimizer'],batches);check();steps[rate]=s
            return steps[rate]
        rate=rate_for_update(arm,c['start_update']+1);step=get_step(rate)
        check();phase('warmup');out['warmup_status']='started'
        warm=step(state['params'],state['optimizer'],batches);jax.block_until_ready(warm);assert_finite(warm)
        out.update(warmup_status='completed',warmup_frame_exposures=count);phase('warmup','completed')
        if state_hashes(state)!=original_hashes:raise ValueError('Warmup mutated parent')
        oracle_candidate=None
        for u in range(c['start_update']+1,c['end_update']+1):
            check();rate=rate_for_update(arm,u);step=get_step(rate)
            if u in (6001,8001):
                phase('oracle',incoming_update=u-1,outgoing_update=u);out['oracle_status']='started'
                oracle_candidate,record=_boundary(state,rate,step,batches,full,model,norm)
                oracles[str(u)]=record;atomic_json(root/f'oracle-{u}.json',record);out['oracle_status']='completed';check()
            if u==c['start_update']+1 or history[-1]['update']==u-1:
                phase('training_block',unknown_if_unreturned=True,possible_first_update=u,
                    possible_last_update=min(x for x in c['score_updates'] if x>=u))
            attempted+=1;pending=True
            pp,oo,mm=step(state['params'],state['optimizer'],batches);jax.block_until_ready((pp,oo,mm));assert_finite((pp,oo,mm))
            candidate={**state,'params':pp,'optimizer':oo,'update':u};validate_state(candidate,norm,u)
            state=candidate;computed+=1;pending=False
            if oracle_candidate is not None:
                _close_trees((state['params'],state['optimizer']),(oracle_candidate['params'],oracle_candidate['optimizer']))
                record=oracles[str(u)];record['committed_state_hashes']=state_hashes(state)
                record['actual_update_verified']=True;atomic_json(root/f'oracle-{u}.json',record);oracle_candidate=None
            if u in c['score_updates']:score(rate)
        check(final=True);references={};phase('reference');out['reference_status']='started'
        for r,p in parts.items():
            e,f=_partition_predictions(state['params'],p,model,c['microbatch_size']);references[f'{r}_energy']=e;references[f'{r}_forces']=f
            out['reference_prediction_frames']+=len(p.record_ids)
        np.savez(root/'reference.npz',**references)
        out['reference_status']='completed'
        contract={'commit':receipt,'config_identity':canonical_hash(c),'arm':arm,'update':int(state['update']),
            'normalization':asdict(norm),'reference_sha256':digest(root/'reference.npz'),
            'provenance_sha256':digest(root/'provenance.json'),'state_hashes':state_hashes(state)}
        atomic_json(root/'checkpoint-contract.json',contract);check(final=True);phase('probe');out['probe_status']='started'
        verified=probe_runner(root,max(.001,arm_deadline-time.perf_counter()))
        out.update(probe_status='completed',checkpoint_verification=verified)
        if verified['status']!='passed' or verified['binding']!=_probe_binding(root):raise ValueError('Child result binding mismatch')
        verify_pins(prepared['source_pins']);verify_code(prepared['code']);check(final=True)
        if state_hashes(prepared['state'])!=original_hashes:raise ValueError('Parent was mutated')
        out.update(status='completed',execution_valid=True,stop_reason='fixed_budget')
    except (Exception,KeyboardInterrupt,SystemExit) as exc:
        out.update(status='incomplete' if isinstance(exc,TimeoutError) else 'failed',execution_valid=False,
            failure_reason=f'{type(exc).__name__}: {exc}',stop_reason='wall_budget' if isinstance(exc,TimeoutError) else 'failure')
        for k in ('warmup_status','oracle_status','probe_status','scoring_status','reference_status'):
            if out[k]=='started':out[k]='unknown'
        if isinstance(exc,subprocess.CalledProcessError):out['probe_stderr']=str(exc.stderr)[-4000:]
    out.update(additional_updates=computed,additional_frame_exposures=computed*count,
        attempted_additional_updates=attempted,pending_update_unknown=pending,
        persisted_additional_updates=(receipt['update']-c['start_update']) if receipt else 0,
        persisted_update=receipt['update'] if receipt else None,history=history,oracles=oracles,
        final_state_hashes=state_hashes(state),parent_state_hashes_after=state_hashes(prepared['state']),
        elapsed_seconds=time.perf_counter()-start,arm_deadline_seconds=c['arm_wall_seconds'])
    phase('arm',out['status'],pending_update_unknown=pending)
    atomic_json(root/'summary.json',out)
    if time.perf_counter()>=arm_deadline:
        out.update(status='incomplete',execution_valid=False,stop_reason='wall_budget',failure_reason='Output persistence exceeded wall cap')
        atomic_json(root/'summary.json',out)
    return out


def _probe_binding(root):
    root=Path(root);receipt=read(root/'commit.json');prov=read(root/'provenance.json')
    return {'config_identity':canonical_hash(read(root/'config.json')),'arm':prov['arm'],
        'provenance_sha256':digest(root/'provenance.json'),'checkpoint_sha256':digest(root/receipt['checkpoint']),
        'history_sha256':digest(root/receipt['history']),'reference_sha256':digest(root/'reference.npz'),
        'input_files_sha256':{r:digest(root/f'{r}-inputs.npz') for r in ('fitting','anchor')},
        'code_identity':canonical_hash(prov['code']),'source_pins_identity':canonical_hash(prov['source_pins'])}


def _probe_saved(root,*,require_parent=False):
    root=Path(root);c=read(root/'config.json');prov=read(root/'provenance.json');contract=read(root/'checkpoint-contract.json');receipt=read(root/'commit.json')
    verify_pins(prov['source_pins']);verify_code(prov['code']);require_backend('cpu')
    if (contract['commit']!=receipt or receipt['update']!=c['end_update']
        or prov['config_identity']!=canonical_hash(c) or contract['config_identity']!=canonical_hash(c)
        or contract['provenance_sha256']!=digest(root/'provenance.json')
        or contract['arm']!=prov['arm'] or contract['update']!=c['end_update']
        or contract['normalization']!=prov['normalization']
        or digest(root/receipt['checkpoint'])!=receipt['checkpoint_sha256']
        or digest(root/receipt['history'])!=receipt['history_sha256']
        or contract['reference_sha256']!=digest(root/'reference.npz')):raise ValueError('Child checkpoint/config binding changed')
    parts=_parts(root,prov);norm=Normalization(**prov['normalization']);model=EnergyModel(ModelConfig(**prov['model']))
    batch=partition_batch(parts['anchor'],np.arange(min(c['microbatch_size'],len(parts['anchor'].record_ids))))
    template=initialize_state(model,batch,norm,c['seed'],.001)
    state=load_checkpoint(root/receipt['checkpoint'],prov,template=template);validate_state(state,norm,c['end_update'])
    if state_hashes(state)!=receipt['state_hashes'] or state_hashes(state)!=contract['state_hashes']:raise ValueError('Restored state identity changed')
    history=read(root/receipt['history'])
    if require_parent:_verify_parent_binding(c,prov,history[0])
    if [r['update'] for r in history]!=c['score_updates'] or history[-1]['state_hashes']!=state_hashes(state):raise ValueError('Saved history contract changed')
    maximum=_prediction_error(state,parts,model,root/'reference.npz',c['microbatch_size'],c['checkpoint_prediction_atol'])
    result={'status':'passed','binding':_probe_binding(root),'update':int(state['update']),
        'adam_count':int(state['optimizer'][0].count),'state_hashes':state_hashes(state),
        'prediction_frames':sum(len(p.record_ids) for p in parts.values()),'maximum_absolute_error':maximum,
        'template_initializations':1,'template_initializer_frames':len(batch['energy']),'precision':'float64',
        'normalization':asdict(norm),'tolerance':c['checkpoint_prediction_atol']}
    atomic_json(root/'checkpoint-verification.json',result)
    return result


def metric_contrast(constant,decay):
    return {'late_decay_minus_constant':decay-constant,
        'late_decay_divided_by_constant':decay/constant if constant else None,
        'ratio_unavailable_reason':None if constant else 'constant denominator is zero'}


def _validated_arm(root,c,*,expected_arm=None,check_parent=True):
    """Reconcile actual saved evidence, independent of summary success flags."""
    root=Path(root);s=read(root/'summary.json');prov=read(root/'provenance.json')
    receipt=read(root/'commit.json');contract=read(root/'checkpoint-contract.json');child=read(root/'checkpoint-verification.json')
    history=read(root/receipt['history']);arm=prov['arm'];verify_code(prov['code']);verify_pins(prov['source_pins'])
    if expected_arm is not None and arm!=expected_arm:raise ValueError('Arm differs from expected schedule name')
    if check_parent:_verify_parent_binding(c,prov,history[0])
    if (read(root/'config.json')!=c or prov['config_identity']!=canonical_hash(c) or contract['config_identity']!=canonical_hash(c)
        or not s['execution_valid'] or s['status']!='completed' or s['arm']!=arm
        or receipt!=contract['commit'] or receipt['update']!=c['end_update']
        or contract['arm']!=arm or contract['update']!=c['end_update'] or contract['normalization']!=prov['normalization']
        or receipt['checkpoint']!=f"snapshots/{c['end_update']}.msgpack" or receipt['history']!=f"histories/{c['end_update']}.json"
        or digest(root/receipt['checkpoint'])!=receipt['checkpoint_sha256'] or digest(root/receipt['history'])!=receipt['history_sha256']
        or contract['provenance_sha256']!=digest(root/'provenance.json') or contract['reference_sha256']!=digest(root/'reference.npz')
        or [r['update'] for r in history]!=c['score_updates'] or s['history']!=history
        or history[0]['state_hashes']!=prov['initial_state_hashes'] or history[-1]['state_hashes']!=receipt['state_hashes']
        or s['initial_state_hashes']!=prov['initial_state_hashes'] or s['parent_state_hashes_after']!=prov['initial_state_hashes']
        or s['final_state_hashes']!=receipt['state_hashes'] or child!=s['checkpoint_verification']
        or child['status']!='passed' or child['binding']!=_probe_binding(root) or child['adam_count']!=c['end_update']
        or child['state_hashes']!=receipt['state_hashes'] or child['maximum_absolute_error']>c['checkpoint_prediction_atol']
        or s['elapsed_seconds']>=c['arm_wall_seconds'] or s['pending_update_unknown']):
        raise ValueError('Saved arm completion bindings disagree')
    state=serialization.msgpack_restore(_read_checkpoint(root/receipt['checkpoint'],prov)['payload'])
    if (state['update']!=c['end_update'] or int(state['optimizer']['0']['count'])!=c['end_update']
        or json.loads(state['normalization_json'])!=prov['normalization'] or state_hashes(state)!=receipt['state_hashes']):
        raise ValueError('Saved checkpoint state metadata disagree')
    parts=_parts(root,prov);count=len(parts['fitting'].record_ids)
    if child['prediction_frames']!=count+len(parts['anchor'].record_ids):raise ValueError('Child frame accounting disagrees')
    prediction_count=count+len(parts['anchor'].record_ids)
    if (s['scoring_status']!='completed' or s['scoring_prediction_frames']!=len(history)*prediction_count
        or s['reference_status']!='completed' or s['reference_prediction_frames']!=prediction_count
        or s['warmup_status']!='completed' or s['warmup_frame_exposures']!=count
        or child['template_initializations']!=1 or child['template_initializer_frames']!=min(c['microbatch_size'],len(parts['anchor'].record_ids))):
        raise ValueError('Prediction/template/warmup work accounting disagrees')
    if check_parent and (child['prediction_frames']!=144 or s['source_prediction_max_error'] is None
        or not np.isfinite(s['source_prediction_max_error']) or s['source_prediction_max_error']>1e-10
        or s['parent_source_reproduction_frames']!=144 or s['parent_endpoint_scoring_frames']!=144
        or s['parent_load_prediction_frames']!=288 or s['template_initializations']!=1 or s['template_initializer_frames']!=16):
        raise ValueError('Production source/child reproduction contract differs')
    additional=c['end_update']-c['start_update']
    if (s['additional_updates']!=additional or s['additional_frame_exposures']!=additional*count
        or s['persisted_additional_updates']!=additional or s['persisted_update']!=c['end_update']
        or s['attempted_additional_updates']!=additional):raise ValueError('New/inherited work accounting disagrees')
    for row in history:
        if row['additional_updates']!=row['update']-c['start_update']:raise ValueError('History additional counter changed')
        if row['update']>c['start_update'] and row['rate']!=rate_for_update(arm,row['update']):raise ValueError('Saved rate observation changed')
        for role in parts:
            if row[role+'_gate']!=gate_result(row['metrics'][role],prov['original_baselines'][role],c['gate_fraction']):
                raise ValueError('Saved acceptance baseline/gate disagrees')
    for u in (6001,8001):
        if c['start_update']<u<=c['end_update']:
            oracle=read(root/f'oracle-{u}.json');incoming=next(r for r in history if r['update']==u-1);outgoing=next(r for r in history if r['update']==u)
            if (oracle['status']!='passed' or not oracle['actual_update_verified'] or not oracle['discarded']
                or oracle['incoming_update']!=u-1 or oracle['outgoing_update']!=u
                or oracle['incoming_state_hashes']!=incoming['state_hashes']
                or oracle['committed_state_hashes']!=outgoing['state_hashes']
                or oracle['proposal_state_hashes']!=outgoing['state_hashes']
                or oracle['rate']!=rate_for_update(arm,u) or s['oracles'].get(str(u))!=oracle
                or oracle['fullbatch_gradient_frames']!=count or oracle['accumulated_gradient_frames']!=count):
                raise ValueError('Boundary oracle/committed state binding disagrees')
    with np.load(root/'reference.npz',allow_pickle=False) as ref:
        residuals={}
        for role,p in parts.items():
            residuals[role]=residual_summary(ref[f'{role}_energy'],p.energy_residual,ref[f'{role}_forces'],p.forces)
            rebuilt=_group_score(p,ref[f'{role}_energy'],ref[f'{role}_forces'])
            for key in MSE_KEYS:
                if not np.isclose(rebuilt[key],history[-1]['metrics'][role][key],atol=1e-12,rtol=1e-10):raise ValueError('Endpoint reference metrics disagree')
            if normalized(rebuilt['per_group'])!=history[-1]['metrics'][role]['per_group']:raise ValueError('Endpoint group metrics disagree')
    end=history[-1];first=history[0]
    return {'status':'completed','execution_valid':True,'arm':arm,'additional_updates':additional,
        'additional_frame_exposures':additional*count,'history':history,'endpoint_metrics':end['metrics'],
        'original_baseline_fractions':{r:end[r+'_gate']['fraction_remaining'] for r in parts},
        'continuation_start_fractions':{r:{k:end['metrics'][r][k]/first['metrics'][r][k] for k in MSE_KEYS} for r in parts},
        'learning_accepted':all(end[r+'_gate']['accepted'] for r in parts),'residual_decomposition':residuals,
        'initial_state_hashes':prov['initial_state_hashes'],'source_prediction_max_error':s['source_prediction_max_error'],
        'child_maximum_prediction_error':child['maximum_absolute_error'],'oracles':s['oracles'],
        'persisted_additional_updates':additional,'verified_persisted_additional_updates':additional,'pending_update_unknown':False,
        'work':{k:v for k,v in s.items() if k in ('warmup_status','warmup_frame_exposures','parent_load_prediction_frames',
            'parent_source_reproduction_frames','parent_endpoint_scoring_frames','template_initializations','template_initializer_frames',
            'scoring_prediction_frames','scoring_status','reference_prediction_frames','reference_status')},
        'child_work':{k:child[k] for k in ('prediction_frames','template_initializations','template_initializer_frames')},
        'elapsed_seconds':s['elapsed_seconds']}


LIMITS=['One selected checkpoint, one seed and fitted supports; no held-out or main-hypothesis result.',
    'The two-stage reduction is one bundled intervention; individual phase effects and optimality are untested.',
    'Fixed finite budget and sparse scores do not prove convergence or identify every spike/mechanism.',
    'Raw energy/force gates use immutable update0 baselines; residual decomposition never recenters them.',
    'Sequential wall times are engineering records, not hardware efficiency rankings.']


def _partial_work(path,c,summary):
    summary=summary or {};work={}
    for k in ('additional_updates','additional_frame_exposures','persisted_additional_updates','attempted_additional_updates'):
        v=summary.get(k);work[k]=v if isinstance(v,int) and not isinstance(v,bool) and v>=0 else None
    verified=None
    try:
        receipt=read(path/'commit.json');hp=path/receipt['history'];cp=path/receipt['checkpoint'];history=read(hp)
        state=serialization.msgpack_restore(_read_checkpoint(cp,read(path/'provenance.json'))['payload'])
        if (digest(hp)!=receipt['history_sha256'] or digest(cp)!=receipt['checkpoint_sha256']
            or state['update']!=receipt['update'] or int(state['optimizer']['0']['count'])!=receipt['update']
            or history[-1]['update']!=receipt['update'] or state_hashes(state)!=receipt['state_hashes']
            or not c['start_update']<=receipt['update']<=c['end_update']):raise ValueError('Partial receipt mismatch')
        verified=receipt['update']-c['start_update']
    except (Exception,KeyboardInterrupt):
        if summary.get('status')=='blocked' and summary.get('additional_updates')==0:verified=0
    work.update(verified_persisted_additional_updates=verified,
        pending_update_unknown=bool(not summary or summary.get('pending_update_unknown') or summary.get('work_since_last_receipt_unknown')),
        work_accounting_status='partial reported computation; persisted count independently reconciled when available')
    if work['additional_updates'] is None and verified is not None:
        work.update(additional_updates=verified,additional_frame_exposures=verified*128,
            work_accounting_status='At least receipt-verified persisted work; exact unreturned computation is unknown')
    return work


def build_report(root,*,allow_pending_finalization=False):
    root=Path(root);errors=[];arms={};c=read(root/'config.json')
    try:
        validate_contract(c)
        if digest(c['source_pins'])!=c['source_pins_sha256']:raise ValueError('Parent pin map changed')
        verify_pins(read(c['source_pins']));verify_code(read(root/'code.json'))
        inventory=read(root/'raw-artifacts.json')
        verify_pins({str(root/name):sha for name,sha in inventory.items()})
    except (Exception,KeyboardInterrupt) as exc:errors.append(f'Global verification: {type(exc).__name__}: {exc}')
    for arm in c['arms']:
        path=root/arm
        try:
            arms[arm]=_validated_arm(path,c,expected_arm=arm)
            if read(path/'provenance.json')['code']!=read(root/'code.json'):raise ValueError('Arm code identity differs from launch')
        except (Exception,KeyboardInterrupt) as exc:
            errors.append(f'{arm}: {type(exc).__name__}: {exc}')
            raw=read(path/'summary.json') if (path/'summary.json').is_file() else None
            arms[arm]={'execution_valid':False,'status':'partial_or_blocked','verification_error':str(exc),
                'raw_summary':raw,**_partial_work(path,c,raw),
                'last_phase':read(path/'phase.json') if (path/'phase.json').is_file() else None}
    valid=not errors and all(a['execution_valid'] for a in arms.values())
    if valid:
        a,b=[arms[n] for n in c['arms']]
        if a['initial_state_hashes']!=b['initial_state_hashes'] or a['history'][0]['metrics']!=b['history'][0]['metrics']:
            errors.append('Arms do not have identical starting state/metrics');valid=False
        completion=read(root/'completion.json')
        if not completion['execution_valid'] or completion['elapsed_seconds']>=c['total_wall_seconds']:
            errors.append('Matrix completion/wall cap invalid');valid=False
        if not allow_pending_finalization:
            try:
                closed=read(root/'finalization.json')
                prefix=Path(c['evidence_log']).read_bytes()[:closed['evidence_prefix_bytes']]
                if (not closed['within_global_cap'] or closed['elapsed_seconds']>=c['total_wall_seconds']
                    or closed['raw_inventory_sha256']!=digest(root/'raw-artifacts.json')
                    or closed['report_sha256']!=digest('reports/late-rate-diagnostics.json')
                    or closed['report_md_sha256']!=digest('reports/late-rate-diagnostics.md')
                    or hashlib.sha256(prefix).hexdigest()!=closed['evidence_prefix_sha256']):
                    raise ValueError('Final report/evidence closure invalid')
            except (Exception,KeyboardInterrupt) as exc:
                errors.append(f'Finalization: {exc}');valid=False
    contrast={r:{k:metric_contrast(arms['constant']['endpoint_metrics'][r][k],arms['late_decay']['endpoint_metrics'][r][k]) for k in MSE_KEYS}
              for r in ('fitting','anchor')} if valid else None
    terminal={}
    if valid:
        for name,a in arms.items():
            old=next(r for r in a['history'] if r['update']==9500)
            terminal[name]={r:{k:(a['endpoint_metrics'][r][k]-old['metrics'][r][k])/old['metrics'][r][k] for k in MSE_KEYS} for r in ('fitting','anchor')}
    return {'status':'completed' if valid else 'partial_or_invalid','execution_valid':valid,'contrasts_available':valid,
        'arms':arms,'endpoint_contrasts':contrast,'terminal_9500_to_10000_relative_changes':terminal,
        'additional_updates':sum(a.get('additional_updates') or 0 for a in arms.values()),
        'additional_frame_exposures':sum(a.get('additional_frame_exposures') or 0 for a in arms.values()),
        'reported_persisted_additional_updates':sum(a.get('persisted_additional_updates') or 0 for a in arms.values()),
        'verified_persisted_additional_updates':sum(a.get('verified_persisted_additional_updates') or 0 for a in arms.values()),
        'any_pending_work_unknown':any(a.get('pending_update_unknown') or a.get('additional_updates') is None for a in arms.values()),
        'work_totals_scope':'Known returned computed counts; partial/unknown work is not complete validated protocol work.',
        'any_learning_accepted':valid and any(a['learning_accepted'] for a in arms.values()),
        'full_pilot':'pending','verification_errors':errors,'limits':LIMITS,
        'raw_inventory_sha256':digest(root/'raw-artifacts.json') if (root/'raw-artifacts.json').is_file() else None}


def evidence(c,identity,title,value):
    with Path(c['evidence_log']).open('a') as f:
        f.write(f'\n## {identity} — {title}\n\n```json\n'+json.dumps(normalized(value),sort_keys=True,indent=2)+'\n```\n')


def _finalize(c,root,start,report):
    """Close after substantive scientific report and evidence writes, without hash cycles."""
    root=Path(root)
    evidence(c,'LATE-RATE-FINALIZATION-001','post-report closure starting',{
        'report_sha256_before_closure':digest('reports/late-rate-diagnostics.json'),
        'status':'finalizing; finalization.json determines cap acceptance','full_pilot':'pending'})
    def invalidate():
        report.update(status='partial_or_invalid',execution_valid=False,contrasts_available=False,endpoint_contrasts=None,any_learning_accepted=False)
        if 'Global cap exceeded during finalization' not in report['verification_errors']:
            report['verification_errors'].append('Global cap exceeded during finalization')
        atomic_json('reports/late-rate-diagnostics.json',report)
        atomic_write('reports/late-rate-diagnostics.md',b'# Late rate diagnostic\n\nFinalization exceeded the global cap; execution invalid and complete contrasts unavailable. Known computed/persisted work remains in JSON. Full pilot pending.\n')
    elapsed=time.perf_counter()-start
    if elapsed>=c['total_wall_seconds']:invalidate()
    prefix=Path(c['evidence_log']).read_bytes()
    closure={'within_global_cap':elapsed<c['total_wall_seconds'],'elapsed_seconds':time.perf_counter()-start,
        'raw_inventory_sha256':digest(root/'raw-artifacts.json') if (root/'raw-artifacts.json').exists() else None,
        'report_sha256':digest('reports/late-rate-diagnostics.json'),'report_md_sha256':digest('reports/late-rate-diagnostics.md'),
        'evidence_prefix_bytes':len(prefix),'evidence_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
        'scope':'After substantive report/evidence persistence; immutable raw inventory and this final marker remain separate.'}
    atomic_json(root/'finalization.json',closure)
    if time.perf_counter()-start>=c['total_wall_seconds'] and closure['within_global_cap']:
        invalidate();closure.update(within_global_cap=False,elapsed_seconds=time.perf_counter()-start,
            report_sha256=digest('reports/late-rate-diagnostics.json'),report_md_sha256=digest('reports/late-rate-diagnostics.md'))
        atomic_json(root/'finalization.json',closure)
    return report


def execute(c):
    validate_contract(c);root=Path(c['run_directory']);root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+c['total_wall_seconds'];results={};failure=None
    atomic_json(root/'config.json',c);code=normalized(code_provenance(Path.cwd()));atomic_json(root/'code.json',code)
    code_root=root/'code-snapshot'
    for name in code['code_files']:atomic_write(code_root/name,Path(name).read_bytes())
    atomic_json(root/'phase.json',{'phase':'parent_preflight','status':'started','unknown_if_unreturned':True})
    evidence(c,'LATE-RATE-PRE-001','once-only production preregistration',{'config':c,'config_identity':canonical_hash(c),
        'code':code,'source_pins_sha256':c['source_pins_sha256'],'limits':LIMITS,'status':'starting'})
    try:
        if digest(c['source_pins'])!=c['source_pins_sha256']:raise ValueError('Parent pin manifest changed')
        verify_pins(read(c['source_pins']))
        atomic_json(root/'saved-observations.json',saved_observations(Path(c['parent_directory'])))
        for arm in c['arms']:
            path=root/arm
            if failure:
                path.mkdir();result={'status':'blocked','execution_valid':False,'failure_reason':failure,
                    'additional_updates':0,'additional_frame_exposures':0,'persisted_additional_updates':0}
                atomic_json(path/'summary.json',result)
            else:
                atomic_json(root/'phase.json',{'phase':'parent_load','arm':arm,'status':'started','unknown_if_unreturned':True})
                prepared=load_parent(c);prepared['code']=code;verify_code(code)
                if time.perf_counter()+c['finalization_reserve_seconds']>=deadline:raise TimeoutError('Matrix reserve exhausted')
                result=_run_arm(c,arm,prepared,path,deadline)
                if not result['execution_valid']:failure=f'{arm}: {result.get("failure_reason","invalid execution")}'
            results[arm]=result;atomic_json(root/'arms.json',results)
            evidence(c,f'LATE-RATE-{arm.upper()}-OUTCOME-001','immediate actual arm outcome',{'root':str(path),
                'summary_sha256':digest(path/'summary.json'),'result':result,'limits':LIMITS})
    except (Exception,KeyboardInterrupt,SystemExit) as exc:
        failure=f'{type(exc).__name__}: {exc}'
        atomic_json(root/'controller-failure.json',{'failure_reason':failure,'completed_arm_names':list(results),
            'last_phase':read(root/'phase.json'),'unreturned_work_unknown':True})
        for arm in c['arms']:
            if arm not in results:
                path=root/arm;path.mkdir(exist_ok=True)
                if (path/'summary.json').exists():result=read(path/'summary.json')
                else:
                    result={'status':'failed_or_blocked','execution_valid':False,'failure_reason':failure,
                        'work_since_last_receipt_unknown':True,'last_phase':read(path/'phase.json') if (path/'phase.json').is_file() else None}
                    atomic_json(path/'summary.json',result)
                results[arm]=result
        atomic_json(root/'arms.json',results)
    elapsed=time.perf_counter()-start
    complete={'execution_valid':failure is None and elapsed<c['total_wall_seconds'],
        'elapsed_seconds':elapsed,'failure_reason':failure,'full_pilot':'pending'}
    atomic_json(root/'completion.json',complete);atomic_json(root/'phase.json',{'phase':'matrix','status':'completed' if complete['execution_valid'] else 'failed'})
    inventory={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='raw-artifacts.json'}
    atomic_json(root/'raw-artifacts.json',inventory)
    if time.perf_counter()>=deadline:
        complete.update(execution_valid=False,elapsed_seconds=time.perf_counter()-start,failure_reason='Final persistence exceeded global cap')
        atomic_json(root/'completion.json',complete);inventory['completion.json']=digest(root/'completion.json');atomic_json(root/'raw-artifacts.json',inventory)
    report=build_report(root,allow_pending_finalization=True);atomic_json('reports/late-rate-diagnostics.json',report)
    lines=['# Late learning-rate continuation diagnostic','',f"Execution valid: {report['execution_valid']}; full pilot: pending.",'',
        '| Arm | Original-baseline fitting E fraction | Original-baseline fitting F fraction | Joint learning gate |','|---|---:|---:|---|']
    for name,a in report['arms'].items():
        if a['execution_valid']:
            frac=a['original_baseline_fractions']['fitting'];lines.append(f"| {name} | {frac[MSE_KEYS[0]]:.8f} | {frac[MSE_KEYS[1]]:.8f} | {a['learning_accepted']} |")
        else:lines.append(f'| {name} | unavailable | unavailable | unavailable |')
    lines+=['',f"Additional verified updates: {report['additional_updates']}; fitting exposures: {report['additional_frame_exposures']}.",'',
        'Contrast orientation: late_decay minus constant; ratios late_decay divided by constant. See JSON for raw units, groups, anchors, histories, residuals, all work records and verification errors.','',*['- '+x for x in LIMITS]]
    atomic_write('reports/late-rate-diagnostics.md',('\n'.join(lines)+'\n').encode())
    evidence(c,'LATE-RATE-REPORT-001','saved-only reconciled report',{'report_sha256':digest('reports/late-rate-diagnostics.json'),
        'report_md_sha256':digest('reports/late-rate-diagnostics.md'),'execution_valid':report['execution_valid'],
        'verification_errors':report['verification_errors'],'additional_updates':report['additional_updates'],
        'additional_frame_exposures':report['additional_frame_exposures'],'endpoint_contrasts':report['endpoint_contrasts'],
        'any_learning_accepted':report['any_learning_accepted'],'full_pilot':'pending','limits':LIMITS})
    return _finalize(c,root,start,report)


def main():
    import argparse
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--config');group.add_argument('--probe');args=p.parse_args()
    if args.probe:
        root=Path(args.probe);validate_contract(read(root/'config.json'));_probe_saved(root,require_parent=True);return 0
    r=execute(read(args.config));return 0 if r['execution_valid'] else 1


if __name__=='__main__':sys.exit(main())
