"""Isolated fresh width/rate diagnostic; immutable scientific controls and honest work receipts."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np
from .width_rate_contract import (CONFIG_ID,PINS_ID,MSE_KEYS,canonical_hash,digest,read,
                                  verify_width_rate_sources,validate_contract,expected_updates,resolve_commit)
from .coverage import take_subset,verify_coverage_subsets
from .coverage_runs import (_arrays,_state_hashes,_partition_predictions,coverage_metrics,gate_result,WallBudget)
from .data.prepare import array_identity,Normalization,PreparedPartition
from .metrics import partition_batch
from .models.backbone import ModelConfig
from .models.energy import EnergyModel,energy_and_forces
from .runs import atomic_write,save_checkpoint,load_checkpoint,assert_finite,require_backend,code_provenance
from .train import initialize_state
from .loss_balance_runs import balance_preflight
from .convergence_runs import verify_convergence_sources


class ExecutionContext:
    """Runtime clocks stay outside the frozen config; actual global time is never double charged."""
    def __init__(self,total_seconds,*,clock=time.perf_counter,global_start=None,arm_start=None,
                 arm_seconds=None,shared_initialization_seconds=0.):
        self.clock=clock;self.global_start=clock() if global_start is None else global_start
        self.global_deadline=self.global_start+total_seconds
        self.arm_start=arm_start;self.arm_seconds=arm_seconds;self.shared_initialization_seconds=shared_initialization_seconds
        self.arm_deadline=None if arm_start is None else arm_start+arm_seconds-shared_initialization_seconds
    def remaining_seconds(self):
        now=self.clock();remaining=self.global_deadline-now
        return remaining if self.arm_deadline is None else min(remaining,self.arm_deadline-now)
    def check_budget(self):
        if self.remaining_seconds()<=0:raise WallBudget('Width/rate boundary wall budget exhausted')
    def for_arm(self,seconds,shared_initialization_seconds):
        return ExecutionContext(self.global_deadline-self.global_start,clock=self.clock,global_start=self.global_start,
                                arm_start=self.clock(),arm_seconds=seconds,shared_initialization_seconds=shared_initialization_seconds)


def atomic_json(path,value):atomic_write(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())

def canonical_state_hashes(state):return _state_hashes({k:jax.tree.map(lambda x:x,state[k]) for k in ('params','optimizer')})


def validate_state(state,norm,update):
    assert_finite(state)
    if int(state['update'])!=update or int(state['optimizer'][0].count)!=update:raise ValueError('State update/Adam count mismatch')
    if json.loads(state['normalization_json'])!=asdict(norm):raise ValueError('State normalization mismatch')
    for x in jax.tree.leaves((state['params'],state['optimizer'][0].mu,state['optimizer'][0].nu)):
        if np.asarray(x).dtype!=np.dtype('float64'):raise ValueError('State float64 mismatch')
    if any(np.any(np.asarray(x)<0) for x in jax.tree.leaves(state['optimizer'][0].nu)):raise ValueError('Negative Adam variance')
    if update==0 and any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves((state['optimizer'][0].mu,state['optimizer'][0].nu))):
        raise ValueError('Fresh Adam moments are not zero')


def parent_operation(config,context,save_partial):
    context.check_budget();start=context.clock()
    record={'operation':'historical_parent_preflight','status':'started','completed_source_prediction_frames':None,
            'historical_template_initializations':None,'elapsed_seconds':0.}
    save_partial(copy.deepcopy(record))
    try:
        result=balance_preflight(config);jax.block_until_ready(result)
        record.update(status='completed',completed_source_prediction_frames=208,historical_template_initializations=1,
                      elapsed_seconds=context.clock()-start)
        save_partial(copy.deepcopy(record));context.check_budget();return result,record
    except BaseException:
        if record['status']!='completed':
            record.update(status='unknown',elapsed_seconds=context.clock()-start)
            save_partial(copy.deepcopy(record))
        raise


def initialize_width_states(config,fitting,anchor,norm,baselines,context,save_partial):
    result={};require_backend('cpu')
    batch=partition_batch(anchor,np.arange(min(config['microbatch_size'],len(anchor.record_ids))))
    for width in (16,32):
        context.check_budget();start=context.clock()
        record={'operation':'shared_training_initialization','features':width,'status':'started','completed_initializations':None}
        save_partial(copy.deepcopy(record))
        model=EnergyModel(ModelConfig(**{**config['model'],'features':width}))
        state=initialize_state(model,batch,norm,config['seed'],.001);jax.block_until_ready(state);validate_state(state,norm,0)
        metrics=coverage_metrics(state['params'],fitting,anchor,model,config['microbatch_size']);assert_finite(metrics)
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:
                if baselines[role][key]<=0 or not np.isclose(metrics[role][key],baselines[role][key],**config['replay_tolerance']):
                    raise ValueError('Fresh zero-head raw baseline changed')
        parameter_count=sum(np.asarray(x).size for x in jax.tree.leaves(state['params']))
        seconds=context.clock()-start
        record.update(status='completed',completed_initializations=1,elapsed_seconds=seconds,
                      parameter_count=parameter_count,initial_state_hashes=canonical_state_hashes(state),
                      initial_metrics=metrics,initialization_batch_record_ids=list(anchor.record_ids)[:config['microbatch_size']],
                      baseline_prediction_frames=len(fitting.record_ids)+len(anchor.record_ids))
        save_partial(copy.deepcopy(record));context.check_budget()
        result[width]={'model':model,'state':state,'initial_metrics':metrics,'parameter_count':parameter_count,
                       'initialization_seconds':seconds,'receipt':record}
    return result


def width_rate_preflight(config,context,save_partial):
    context.check_budget();sources=verify_width_rate_sources(config);context.check_budget()
    # Static existing source/control checks see the unmodified historical config.
    historical=sources['historical_config'];verify_convergence_sources(historical);context.check_budget()
    parent,record=parent_operation(historical,context,save_partial)
    train,manifest,norm,selected,ancestry,calibration=parent
    planned=read(config['subsets_manifest']);verify_coverage_subsets(selected,planned)
    if manifest['identity']!=config['expected_prepared_identity'] or manifest['split_hash']!=config['expected_split_hash'] or planned['identity']!=config['expected_subsets_identity']:
        raise ValueError('Prepared/subset identity changed')
    fitting=take_subset(train,selected['subsets']['diverse128']['record_ids']);anchor=take_subset(train,selected['subsets']['narrow16']['record_ids'])
    if (len(fitting.record_ids),len(anchor.record_ids))!=(128,16) or not set(anchor.record_ids)<=set(fitting.record_ids):raise ValueError('Frozen fitted support mismatch')
    inputs={role:{'arrays':{k:array_identity(v) for k,v in _arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships}
            for role,p in [('fitting',fitting),('anchor',anchor)]}
    if inputs!=sources['historical_inputs']:raise ValueError('Prepared inputs differ from original balanced control')
    widths=initialize_width_states(config,fitting,anchor,norm,sources['original_baselines'],context,save_partial)
    if _state_hashes(widths[16]['state'])!=sources['historical_initial_state_hashes']:raise ValueError('Fresh16 initialization differs from historical')
    verify_width_rate_sources(config);context.check_budget()
    return {'fitting':fitting,'anchor':anchor,'normalization':norm,'widths':widths,'sources':sources,'parent_work':record,'inputs':inputs}

# Numerical helpers are private to the reviewed production wrapper; tiny engineering
# fixtures call the engine directly and cannot enter the CLI's frozen-config route.
from .loss_balance import build_weighted_step
import subprocess
import sys


def independent_adam_step(params,optimizer,gradient,learning_rate):
    """NumPy bias-corrected Adam, independent of Optax's update transformation."""
    current=optimizer[0];count=int(current.count)+1
    mu=jax.tree.map(lambda m,g:.9*np.asarray(m)+.1*np.asarray(g),current.mu,gradient)
    nu=jax.tree.map(lambda v,g:.999*np.asarray(v)+.001*np.asarray(g)**2,current.nu,gradient)
    updated=jax.tree.map(lambda p,m,v:np.asarray(p)-learning_rate*(m/(1-.9**count))/(np.sqrt(v/(1-.999**count))+1e-8),params,mu,nu)
    nextopt=(current._replace(count=np.asarray(count,dtype=np.int32),mu=mu,nu=nu),*optimizer[1:])
    return updated,nextopt


def _trees_close(actual,wanted):
    a,astr=jax.tree.flatten(actual);b,bstr=jax.tree.flatten(wanted)
    if astr!=bstr:raise ValueError('Oracle state tree structure mismatch')
    maximum=0.
    for left,right in zip(a,b,strict=True):
        left,right=np.asarray(left),np.asarray(right)
        if left.shape!=right.shape or not np.allclose(left,right,atol=1e-10,rtol=1e-8):raise ValueError('Independent Adam oracle mismatch')
        maximum=max(maximum,float(np.max(np.abs(left-right))))
    return maximum


def _tolerance_fraction(actual,wanted):
    fractions=[float(np.max(np.abs(np.asarray(a)-np.asarray(b))/(1e-10+1e-8*np.abs(np.asarray(b)))))
               for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(wanted),strict=True)]
    return max(fractions,default=0.)


def first_step_oracle(state,batches,fullbatch,model,norm,rate,weights,context):
    context.check_budget();start=context.clock();validate_state(state,norm,0);before=canonical_state_hashes(state)
    def loss(params):
        energy,forces=energy_and_forces(params,fullbatch,model=model)
        return (weights['energy']*jax.numpy.mean((energy-fullbatch['energy'])**2)/norm.energy_loss_scale_eV**2+
                weights['force']*jax.numpy.mean((forces-fullbatch['forces'])**2)/norm.force_loss_scale_eV_per_A**2)
    # One full fitting batch; never call the accumulated gradient builder for the reference.
    gradient=jax.jit(jax.grad(loss))(state['params']);jax.block_until_ready(gradient);assert_finite(gradient);context.check_budget()
    rp,ro=independent_adam_step(state['params'],state['optimizer'],gradient,rate)
    step=build_weighted_step(model,norm,rate,weights);step.prepare(state['params'],state['optimizer'],batches);context.check_budget()
    ap,ao,_=step(state['params'],state['optimizer'],batches);jax.block_until_ready((ap,ao));assert_finite((ap,ao));context.check_budget()
    discrepancy=_trees_close((ap,ao),(rp,ro))
    predicted=energy_and_forces(ap,fullbatch,model=model);reference=energy_and_forces(rp,fullbatch,model=model)
    jax.block_until_ready((predicted,reference));prediction_error=_trees_close(predicted,reference);context.check_budget()
    candidate={**state,'params':ap,'optimizer':ao,'update':1};validate_state(candidate,norm,1)
    if canonical_state_hashes(state)!=before:raise ValueError('Oracle mutated fresh initial state')
    return {'state':candidate,'reference_params':rp,'gradient':gradient,
            'receipt':{'status':'passed','count':1,'state_maximum_absolute_difference':discrepancy,
                'prediction_maximum_absolute_difference':prediction_error,
                'state_maximum_tolerance_fraction':_tolerance_fraction((ap,ao),(rp,ro)),
                'prediction_maximum_tolerance_fraction':_tolerance_fraction(predicted,reference),'atol':1e-10,'rtol':1e-8,
                'fullbatch_reference_frames':len(fullbatch['energy']),'accumulated_proposal_frames':sum(len(b['energy']) for b in batches),
                'prediction_reference_frames':2*len(fullbatch['energy']),'discarded':True,
                'initial_state_hashes':before,'proposed_state_hashes':canonical_state_hashes(candidate),
                'gradient_sha256':canonical_hash({str(i):array_identity(np.asarray(x)) for i,x in enumerate(jax.tree.leaves(gradient))}),
                'elapsed_seconds':context.clock()-start}}


def commit_state(root,state,history,provenance,norm):
    """Publish the receipt last; never overwrite a version even if it is an orphan."""
    root=Path(root);update=int(state['update']);validate_state(state,norm,update)
    if not history or history[-1]['update']!=update:raise ValueError('State/history endpoint mismatch')
    cp=root/'snapshots'/f'{update}.msgpack';hp=root/'histories'/f'{update}.json'
    if cp.exists() or hp.exists():raise ValueError('Immutable version already exists')
    cp.parent.mkdir(parents=True,exist_ok=True);hp.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(cp,state,provenance);atomic_json(hp,history)
    receipt={'update':update,'checkpoint':str(cp.relative_to(root)),'history':str(hp.relative_to(root)),
             'checkpoint_sha256':digest(cp),'history_sha256':digest(hp),'state_hashes':canonical_state_hashes(state)}
    atomic_json(root/'commit.json',receipt);return receipt


def _row(state,metrics,config,norm,baselines,frame_count):
    update=int(state['update']);row={'update':update,'frame_exposures':update*frame_count,'metrics':metrics,'weighted_loss':{},'original_baseline_fraction':{}}
    for role,label in [('fitting','fit_gate'),('anchor','anchor_gate')]:
        gate=gate_result(metrics[role],baselines[role],config['gate_fraction']);row[label]=gate
        row['original_baseline_fraction'][role]=gate['fraction_remaining']
        row['weighted_loss'][role]=(config['loss_weights']['energy']*metrics[role][MSE_KEYS[0]]/norm.energy_loss_scale_eV**2+
                                    config['loss_weights']['force']*metrics[role][MSE_KEYS[1]]/norm.force_loss_scale_eV_per_A**2)
    return row


def _replay(metrics,wanted,tolerance):
    maxerror=0.
    for role in ('fitting','anchor'):
        for key in MSE_KEYS:
            a,b=metrics[role][key],wanted[role][key]
            if not np.isclose(a,b,**tolerance):raise ValueError('Historical replay raw MSE mismatch')
            maxerror=max(maxerror,abs(a-b))
    return maxerror


def _launch_probe(root,context):
    context.check_budget();timeout=context.remaining_seconds()
    if timeout<=0:raise WallBudget('No probe budget remains')
    subprocess.run([sys.executable,'-m','force_consistency.width_rate_runs','--probe',str(root.resolve())],
                   check=True,capture_output=True,timeout=timeout)
    context.check_budget();return read(root/'checkpoint-verification.json')


def _run_arm(config,case,initial,fitting,anchor,model,normalization,original_baselines,historical,root,provenance,context,*,probe_runner=_launch_probe):
    root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);start=context.arm_start if context.arm_start is not None else context.clock();state=copy.deepcopy(initial);norm=normalization
    before=canonical_state_hashes(initial);fit_count=len(fitting.record_ids);anchor_count=len(anchor.record_ids)
    inputs={role:{'arrays':{k:array_identity(v) for k,v in _arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships}
            for role,p in [('fitting',fitting),('anchor',anchor)]}
    prov={**provenance,'config_identity':canonical_hash(config),'case':case,'normalization':asdict(norm),'original_baselines':original_baselines,
          'inputs':inputs,'initial_state_hashes':before,'precision':'float64','backend':'cpu','model':asdict(model.config)}
    # JSON stringifies nested integer keys. Normalize once before both the metadata
    # file and strict checkpoint envelope serialization, preserving their identity.
    prov=json.loads(json.dumps(prov,allow_nan=False))
    history=[];times=[];evaluation_times=[];committed=None;replayed=[];replay_error=0.;probe_start=None
    out={'status':'failed','executed':True,'execution_valid':False,'full_pilot':'pending','case':case,'updates':0,'attempted_updates':0,
         'persisted_update':None,'pending_update_unknown':False,'frame_exposures':0,'history':history,'original_baselines':original_baselines,
         'warmup_frame_exposures':0,'warmup_work_status':'not_started','oracle_work_status':'not_started','oracle_verified':False,
         'probe_work_status':'not_started','checkpoint_new_process_verified':False,'source_hashes_unchanged':False,'provenance':prov,'initial_state_hashes':before,'initial_raw_state_hashes':_state_hashes(initial),
         'parameter_count':sum(np.asarray(x).size for x in jax.tree.leaves(initial['params'])),'fitting_frame_count':fit_count,'anchor_frame_count':anchor_count,
         'shared_initialization_seconds':context.shared_initialization_seconds,'compile_seconds':0.,'first_step_reference_seconds':0.,'warmup_seconds':0.,'probe_seconds':0.,'checkpoint_seconds':0.}
    def score():
        nonlocal committed,replay_error
        context.check_budget();t=context.clock();metrics=coverage_metrics(state['params'],fitting,anchor,model,config['microbatch_size']);assert_finite(metrics)
        evaluation_times.append(context.clock()-t);context.check_budget()
        row=_row(state,metrics,config,norm,original_baselines,fit_count);history.append(row)
        t=context.clock();committed=commit_state(root,state,history,prov,norm);out['checkpoint_seconds']+=context.clock()-t;out['persisted_update']=committed['update'];context.check_budget()
        if int(state['update']) in historical:
            replay_error=max(replay_error,_replay(metrics,historical[int(state['update'])],config['replay_tolerance']));replayed.append(int(state['update']))
        return metrics
    try:
        context.check_budget();validate_state(state,norm,0)
        if asdict(model.config)!={**config['model'],'features':case['features']}:raise ValueError('Cell model features/config mismatch')
        atomic_json(root/'config.json',config);atomic_json(root/'case.json',case);atomic_json(root/'provenance.json',prov)
        for role,p in [('fitting',fitting),('anchor',anchor)]:np.savez(root/f'{role}-inputs.npz',**_arrays(p))
        prov['input_files_sha256']={role:digest(root/f'{role}-inputs.npz') for role in ('fitting','anchor')}
        atomic_json(root/'provenance.json',prov)
        metrics=score()
        batches=[partition_batch(fitting,np.arange(i,min(i+config['microbatch_size'],fit_count))) for i in range(0,fit_count,config['microbatch_size'])]
        full=partition_batch(fitting,np.arange(fit_count));context.check_budget();out['oracle_work_status']='started'
        oracle=first_step_oracle(state,batches,full,model,norm,case['learning_rate'],config['loss_weights'],context)
        out.update(oracle_work_status='completed',oracle_verified=True,oracle_verification=oracle['receipt'],first_step_reference_seconds=oracle['receipt']['elapsed_seconds'])
        oracle_metrics=coverage_metrics(oracle['state']['params'],fitting,anchor,model,config['microbatch_size']);context.check_budget()
        out['oracle_metrics']=oracle_metrics;out['oracle_metrics_prediction_frames']=fit_count+anchor_count;atomic_json(root/'oracle-verification.json',{**oracle['receipt'],'metrics':oracle_metrics})
        step=build_weighted_step(model,norm,case['learning_rate'],config['loss_weights']);t=context.clock();context.check_budget()
        step.prepare(state['params'],state['optimizer'],batches);context.check_budget();out['compile_seconds']=context.clock()-t
        t=context.clock();out['warmup_work_status']='started';warm=step(state['params'],state['optimizer'],batches);jax.block_until_ready(warm);assert_finite(warm)
        out.update(warmup_frame_exposures=fit_count,warmup_work_status='completed',warmup_seconds=context.clock()-t,
                   warmup_source_unchanged=canonical_state_hashes(state)==before);context.check_budget()
        if not out['warmup_source_unchanged']:raise ValueError('Warmup advanced fresh state')
        for update in range(1,config['max_updates']+1):
            context.check_budget();out['attempted_updates']+=1;out['pending_update_unknown']=True;t=context.clock()
            pp,oo,mm=step(state['params'],state['optimizer'],batches);jax.block_until_ready((pp,oo,mm));assert_finite((pp,oo,mm))
            candidate={**state,'params':pp,'optimizer':oo,'update':update};validate_state(candidate,norm,update)
            state=candidate;times.append(context.clock()-t);out.update(updates=update,pending_update_unknown=False);context.check_budget()
            if update in config['score_updates']:
                metrics=score()
                if update==1:_replay(metrics,oracle_metrics,config['replay_tolerance'])
        context.check_budget();references={}
        for role,p in [('fitting',fitting),('anchor',anchor)]:
            context.check_budget();e,f=_partition_predictions(state['params'],p,model,config['microbatch_size']);references[f'{role}_energy']=e;references[f'{role}_forces']=f
        np.savez(root/'reference.npz',**references);out['reference_prediction_frames']=fit_count+anchor_count;context.check_budget()
        contract={'case':case,'update':config['max_updates'],'normalization':asdict(norm),'state_hashes':canonical_state_hashes(state),
                  'commit':committed,'inputs':inputs,'input_files_sha256':prov['input_files_sha256'],'reference_sha256':digest(root/'reference.npz'),'config_identity':canonical_hash(config)}
        atomic_json(root/'checkpoint-contract.json',contract);context.check_budget();t=context.clock()
        probe_start=t;out['probe_work_status']='started'
        receipt=probe_runner(root,context);out.update(probe_seconds=context.clock()-probe_start,probe_work_status='completed');context.check_budget()
        if (receipt.get('status')!='passed' or receipt.get('update')!=config['max_updates'] or receipt.get('adam_count')!=config['max_updates']
            or receipt.get('state_hashes')!=contract['state_hashes'] or receipt.get('checkpoint_sha256')!=committed['checkpoint_sha256']
            or receipt.get('normalization')!=asdict(norm) or receipt.get('case')!=case or receipt.get('config_identity')!=canonical_hash(config)):
            raise ValueError('Fresh subprocess receipt mismatch')
        out.update(status='completed',execution_valid=True,stop_reason='fixed_budget',checkpoint_new_process_verified=True,
                   checkpoint_verification=receipt,checkpoint_sha256=committed['checkpoint_sha256'],reference_prediction_frames=fit_count+anchor_count)
    except BaseException as exc:
        if not isinstance(exc,(Exception,KeyboardInterrupt,SystemExit)):raise
        out.update(status='incomplete' if isinstance(exc,(WallBudget,subprocess.TimeoutExpired)) else 'failed',execution_valid=False,
                   failure_reason=f'{type(exc).__name__}: {exc}',stop_reason='wall_budget' if isinstance(exc,(WallBudget,subprocess.TimeoutExpired)) else 'failure')
        if out['probe_work_status']=='started':out.update(probe_work_status='unknown',probe_seconds=context.clock()-probe_start)
        if isinstance(exc,subprocess.CalledProcessError):out['probe_stderr']=(exc.stderr.decode() if isinstance(exc.stderr,bytes) else str(exc.stderr))[-4000:]
        for field in ('oracle_work_status','warmup_work_status'):
            if out[field]=='started':out[field]='unknown'
        # Never publish a newer unscored state or overwrite a committed version in failure handling.
    out.update(frame_exposures=out['updates']*fit_count,persisted_frame_exposures=(out['persisted_update'] or 0)*fit_count,
               final_state_hashes=canonical_state_hashes(state),initial_state_hashes_after=canonical_state_hashes(initial),
               evaluation_passes=len(evaluation_times),evaluation_frame_count=len(evaluation_times)*(fit_count+anchor_count),
               steady_update_seconds_total=sum(times),steady_update_seconds_mean=float(np.mean(times)) if times else None,
               evaluation_seconds_total=sum(evaluation_times),replay_compared_updates=replayed,replay_maximum_absolute_mse_difference=replay_error,
               replay_verified=not historical or replayed==sorted(historical),elapsed_seconds=context.clock()-start)
    out['charged_elapsed_seconds']=out['elapsed_seconds']+out['shared_initialization_seconds']
    if history:out.update(final_metrics=history[-1]['metrics'],fit_gate=history[-1]['fit_gate'],anchor_gate=history[-1]['anchor_gate'])
    if out['initial_state_hashes_after']!=before:out.update(status='failed',execution_valid=False,failure_reason='Shared initial state changed')
    if out['execution_valid'] and (not out['replay_verified'] or out['charged_elapsed_seconds']>=case['wall_seconds'] or context.remaining_seconds()<=0):
        out.update(status='failed',execution_valid=False,failure_reason='Replay or output budget failed')
    try:
        atomic_json(root/'summary.json',out)
        context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,final_persistence_failure=f'{type(exc).__name__}: {exc}')
        atomic_write(root/'failure.json',(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    return out


def run_width_rate_arm(config,case,initial,fitting,anchor,model,normalization,original_baselines,historical,root,provenance,context):
    validate_contract(config)
    if case not in config['cases']:raise ValueError('Unreviewed cell')
    sources=verify_width_rate_sources(config)
    if original_baselines!=sources['original_baselines']:raise ValueError('Copied baseline changed')
    if (len(fitting.record_ids),len(anchor.record_ids))!=(128,16):raise ValueError('Unreviewed support')
    if case['name']=='width16-lr1e-3':
        if historical!=sources['historical']:raise ValueError('Replay history changed')
    elif historical:raise ValueError('Historical oracle applied to wrong cell')
    out=_run_arm(config,case,initial,fitting,anchor,model,normalization,original_baselines,historical,root,provenance,context)
    try:verify_width_rate_sources(config);context.check_budget();out['source_hashes_unchanged']=True
    except BaseException as exc:out.update(status='failed',execution_valid=False,failure_reason=f'Source/budget: {type(exc).__name__}: {exc}')
    out['elapsed_seconds']=context.clock()-(context.arm_start if context.arm_start is not None else context.global_start)
    out['charged_elapsed_seconds']=out['elapsed_seconds']+context.shared_initialization_seconds
    try:
        atomic_json(Path(root)/'summary.json',out);context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,final_persistence_failure=f'{type(exc).__name__}: {exc}')
        atomic_write(Path(root)/'failure.json',(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    return out


def _probe_saved(root):
    root=Path(root);config=read(root/'config.json');case=read(root/'case.json');prov=read(root/'provenance.json');contract=read(root/'checkpoint-contract.json')
    if case!=prov['case'] or contract['case']!=case or contract['config_identity']!=canonical_hash(config) or prov['config_identity']!=canonical_hash(config):raise ValueError('Probe config/case identity mismatch')
    model=EnergyModel(ModelConfig(**{**config['model'],'features':case['features']}));norm=Normalization(**prov['normalization'])
    if asdict(model.config)!=prov['model']:raise ValueError('Probe feature template mismatch')
    parts={}
    for role in ('fitting','anchor'):
        meta=prov['inputs'][role]
        with np.load(root/f'{role}-inputs.npz',allow_pickle=False) as data:
            if set(data.files)!=set(meta['arrays']) or any(array_identity(data[k])!=meta['arrays'][k] for k in data.files):raise ValueError('Probe input arrays mismatch')
            parts[role]=PreparedPartition(**{k:np.array(data[k]) for k in data.files},record_ids=tuple(meta['record_ids']),memberships=meta['memberships'])
    if contract['normalization']!=asdict(norm) or contract['input_files_sha256']!=prov['input_files_sha256'] or any(digest(root/f'{r}-inputs.npz')!=prov['input_files_sha256'][r] for r in ('fitting','anchor')):raise ValueError('Probe physical input-file/normalization contract mismatch')
    if contract['inputs']!=prov['inputs'] or contract['reference_sha256']!=digest(root/'reference.npz'):raise ValueError('Probe reference/input identity mismatch')
    anchor=parts['anchor'];batch=partition_batch(anchor,np.arange(min(config['microbatch_size'],len(anchor.record_ids))))
    template=initialize_state(model,batch,norm,config['seed'],case['learning_rate']);receipt,history=resolve_commit(root)
    if receipt!=contract['commit'] or receipt['update']!=config['max_updates']:raise ValueError('Probe commit contract mismatch')
    state=load_checkpoint(root/receipt['checkpoint'],prov,template=template);validate_state(state,norm,config['max_updates']);hashes=canonical_state_hashes(state)
    if hashes!=receipt['state_hashes'] or hashes!=contract['state_hashes']:raise ValueError('Probe state identity mismatch')
    maximum=0.
    with np.load(root/'reference.npz',allow_pickle=False) as reference:
        if set(reference.files)!= {'fitting_energy','fitting_forces','anchor_energy','anchor_forces'}:raise ValueError('Probe reference fields mismatch')
        for role,part in parts.items():
            energy,forces=_partition_predictions(state['params'],part,model,config['microbatch_size'])
            for label,actual in [('energy',energy),('forces',forces)]:
                wanted=reference[f'{role}_{label}'];assert_finite(wanted)
                if wanted.shape!=actual.shape or not np.allclose(actual,wanted,atol=1e-10,rtol=0):raise ValueError('Probe predictions mismatch')
                maximum=max(maximum,float(np.max(abs(actual-wanted))))
    result={'status':'passed','update':int(state['update']),'adam_count':int(state['optimizer'][0].count),'normalization':asdict(norm),
            'state_hashes':hashes,'checkpoint_sha256':receipt['checkpoint_sha256'],'history_sha256':receipt['history_sha256'],
            'case':case,'config_identity':canonical_hash(config),'fitting_record_count':len(parts['fitting'].record_ids),
            'anchor_record_count':len(anchor.record_ids),'maximum_absolute_error':maximum,'tolerance':1e-10,'precision':'float64','template_initializations':1,'template_initializer_frames':min(config['microbatch_size'],len(anchor.record_ids)),
            'prediction_frames':len(parts['fitting'].record_ids)+len(anchor.record_ids)}
    atomic_json(root/'checkpoint-verification.json',result);return result


def width_rate_probe(root):
    config=read(Path(root)/'config.json');validate_contract(config)
    if read(Path(root)/'case.json') not in config['cases']:raise ValueError('Unreviewed probe cell')
    return _probe_saved(root)


def execute_width_rate_matrix(root,config,runner,verify_sources,context):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any((root/c['name']).exists() for c in config['cases']):raise ValueError('Cell directory already exists')
    out={'status':'failed','execution_valid':False,'full_pilot':'pending','arms':{}};blocked=None
    for case in config['cases']:
        name=case['name']
        if blocked is None:
            try:context.check_budget();verify_sources()
            except BaseException as exc:blocked=f'{type(exc).__name__}: {exc}'
        if blocked is not None:arm={'status':'blocked','executed':False,'execution_valid':False,'case':case,'stop_reason':blocked}
        else:
            arm=None
            try:
                arm=runner(case);verify_sources();context.check_budget()
                if not arm['execution_valid']:blocked=name+'_execution_failure'
            except BaseException as exc:
                if arm is None:
                    child=root/name
                    arm=read(child/'failure.json') if (child/'failure.json').exists() else read(child/'summary.json') if (child/'summary.json').exists() else {'executed':True,'case':case}
                arm.update(status='failed',execution_valid=False,failure_reason=f'{type(exc).__name__}: {exc}');blocked=arm['failure_reason']
        out['arms'][name]=arm
        try:atomic_json(root/'summary.json',out)
        except BaseException as exc:
            blocked=f'Controller persistence: {type(exc).__name__}: {exc}';out['failure_reason']=blocked
            atomic_write(root/'controller-failure.json',(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    out.update(status='completed' if blocked is None else 'failed',execution_valid=blocked is None,elapsed_seconds=context.clock()-context.global_start)
    try:context.check_budget();verify_sources();atomic_json(root/'summary.json',out);context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,failure_reason=f'{type(exc).__name__}: {exc}')
        atomic_write(root/'controller-failure.json',(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    return out


def _evidence(config,identity,title,details):
    with Path(config['evidence_log']).open('a') as f:
        f.write('\n## '+identity+' — '+title+'\n\nFrozen one-seed width/rate sensitivity; fitted anchors, full pilot pending.\n\n```json\n'+json.dumps(details,indent=2,sort_keys=True,allow_nan=False)+'\n```\n')


def diagnose_width_rate(config):
    validate_contract(config);root=Path(config['run_directory'])
    if root.exists():raise ValueError('Protected production root already exists')
    context=ExecutionContext(config['total_wall_seconds']);root.mkdir(parents=True)
    _evidence(config,'WIDTHRATE-PRE-001','fixed matrix preregistered before numerical work',{'config_identity':CONFIG_ID,'config':config,
        'acceptance':'Original strict joint E/F<.1 raw initial MSE on both fitting and fully fitted anchors; no pilot clearance',
        'rationale':'Fresh width/rate sensitivity after all original6000-update continuations failed joint gates; 16/.001 is numerical replay control'})
    records=[]
    def partial(record):
        records.append(copy.deepcopy(record));atomic_json(root/'preflight-work.json',records)
    try:
        atomic_json(root/'config.json',config);prepared=width_rate_preflight(config,context,partial)
        sources=prepared['sources'];atomic_json(root/'sources.json',sources)
        provenance={**code_provenance(Path.cwd()),'prepared_identity':config['expected_prepared_identity'],'split_hash':config['expected_split_hash'],
                    'subsets_identity':config['expected_subsets_identity'],'source_pin_manifest_identity':PINS_ID,'sources':sources,
                    'new_code_sha256':{str(p):digest(p) for p in [Path(__file__),Path('src/force_consistency/width_rate_contract.py'),Path('src/force_consistency/cli.py')]}}
        atomic_json(root/'provenance.json',provenance)
        for width,entry in prepared['widths'].items():
            save_checkpoint(root/f'initial-width{width}.msgpack',entry['state'],{'features':width,'normalization':asdict(prepared['normalization']),'state_hashes':canonical_state_hashes(entry['state'])})
        def runner(case):
            entry=prepared['widths'][case['features']];armcontext=context.for_arm(case['wall_seconds'],entry['initialization_seconds'])
            _evidence(config,'WIDTHRATE-'+case['name']+'-PRE','cell starts',{'case':case,'initial_state_hashes':canonical_state_hashes(entry['state']),'fresh_count':0})
            arm=run_width_rate_arm(config,case,entry['state'],prepared['fitting'],prepared['anchor'],entry['model'],prepared['normalization'],
                    sources['original_baselines'],sources['historical'] if case['name']=='width16-lr1e-3' else {},root/case['name'],provenance,armcontext)
            _evidence(config,'WIDTHRATE-'+case['name']+'-RESULT','cell result retained',{'status':arm['status'],'execution_valid':arm['execution_valid'],
                'updates':arm.get('updates'),'persisted_update':arm.get('persisted_update'),'attempted_updates':arm.get('attempted_updates'),
                'failure_reason':arm.get('failure_reason'),'fit_gate':arm.get('fit_gate'),'anchor_gate':arm.get('anchor_gate'),
                'summary_sha256':digest(root/case['name']/'summary.json')})
            return arm
        out=execute_width_rate_matrix(root,config,runner,lambda:verify_width_rate_sources(config),context)
        out['parent_work']=prepared['parent_work'];out['shared_initializations']={str(k):v['receipt'] for k,v in prepared['widths'].items()}
    except BaseException as exc:
        out={'status':'failed','execution_valid':False,'full_pilot':'pending','failure_reason':f'Preflight: {type(exc).__name__}: {exc}',
             'preflight_work':records,'arms':{c['name']:{'case':c,'status':'blocked','executed':False,'execution_valid':False,'stop_reason':'preflight_failure'} for c in config['cases']}}
    out['elapsed_seconds']=context.clock()-context.global_start
    try:
        context.check_budget();atomic_json(root/'summary.json',out);context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,final_output_failure=f'{type(exc).__name__}: {exc}')
        atomic_write(root/'controller-failure.json',(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    _evidence(config,'WIDTHRATE-RESULT-001','matrix stopped; every cell retained',{'status':out['status'],'execution_valid':out['execution_valid'],
        'elapsed_seconds':out['elapsed_seconds'],'failure_reason':out.get('failure_reason'),'arms':{k:v['status'] for k,v in out['arms'].items()},
        'final_receipt':str(root/'controller-failure.json' if (root/'controller-failure.json').exists() else root/'summary.json')})
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',type=Path,required=True);args=parser.parse_args();width_rate_probe(args.probe)
