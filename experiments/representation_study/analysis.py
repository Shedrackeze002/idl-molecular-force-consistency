"""Independent endpoint scoring and fixed-window paired descriptive summaries."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import jax
import numpy as np
from force_consistency.data.prepare import load_training_bundle
from force_consistency.data.splits import canonical_hash
from force_consistency.metrics import partition_batch
from force_consistency.runs import assert_finite,code_provenance,load_checkpoint
from force_consistency.train import RunConfig,model_and_predictor,initialize_state,initialization_identity,sample_training_batch
from .runner import ROOT,read_json,write_json,inspect_checkpoint,IdentityError

# Reuse the preserved independent metric reducer without editing its source.
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_e3x_adequacy import prediction_metrics


def history_metrics(history,target=200000):
    if [x['update'] for x in history]!=list(range(0,target+1,500)):
        raise ValueError('Incomplete or duplicate validation schedule')
    values=[x['equal_connection_force_mae_eV_per_A'] for x in history]
    if not all(np.isfinite(x) and x>=0 for x in values):raise ValueError('Invalid validation scores')
    i=min(range(len(values)),key=values.__getitem__)
    late=[x for x in history if 190500<=x['update']<=200000]
    if target==200000 and len(late)!=20:raise ValueError('Missing late20 events')
    return {'best_update':history[i]['update'],'best_equal_connection_force_mae_eV_per_A':values[i],
        'last_equal_connection_force_mae_eV_per_A':values[-1],
        'late20_equal_connection_force_mae_eV_per_A':float(np.mean([x['equal_connection_force_mae_eV_per_A'] for x in late])) if late else None,
        'late20_updates':[x['update'] for x in late]}


def paired_summary(rows):
    values={x['candidate']:x for x in rows}
    if len(values)!=len(rows) or not all(f'E64-{seed}' in values for seed in (0,1)):raise ValueError('Missing/duplicate reference pair')
    out={}
    for name in sorted({k.rsplit('-',1)[0] for k in values}-{'E64'}):
        if not all(f'{name}-{seed}' in values for seed in (0,1)):raise ValueError('Missing scalar seed')
        best=[values[f'{name}-{s}']['best']-values[f'E64-{s}']['best'] for s in (0,1)]
        late=[values[f'{name}-{s}']['late20']-values[f'E64-{s}']['late20'] for s in (0,1)]
        bm=float(np.mean(best));lm=float(np.mean(late))
        out[name]={'best_seed_deltas':best,'late20_seed_deltas':late,'best_mean_delta':bm,'late20_mean_delta':lm,
            'interpretation':'mixed' if bm*lm<0 or any(a*b<0 for a,b in zip(best,late)) else 'descriptive fixed-recipe comparison',
            'sign':'scalar minus E64; positive favors E64; no statistical superiority claim'}
    return out


def segment_timings(run_directory):
    run=Path(run_directory);rows=[]
    for path in (sorted(run.parent.glob('[0-9]*')) if run.parent.name=='segments' else [run]):
        try:
            saved=read_json(path/'summary.json')
            if not isinstance(saved,dict) or saved.get('elapsed_seconds') is None:raise ValueError('Invalid prior timing summary')
            timing_keys=('elapsed_seconds','steady_update_seconds_mean','validation_seconds_total','compilation_seconds','warmup_execution_seconds')
            if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0) for v in (saved.get(k) for k in timing_keys)):raise ValueError('Invalid prior timing field')
            rows.append({'segment':path.name,'timing_status':'recorded',
                **{key:saved.get(key) for key in ('elapsed_seconds','steady_update_seconds_mean','validation_seconds_total','compilation_seconds','warmup_execution_seconds')}})
        except (OSError,ValueError,TypeError) as exc:
            rows.append({'segment':path.name,'timing_status':'unknown','elapsed_seconds':None,'reason':f'{type(exc).__name__}: {exc}'})
    return rows


def score_run(run_directory,output):
    run=Path(run_directory);output=Path(output);prov=read_json(run/'provenance.json');cfg=RunConfig(**read_json(run/'config.json'))
    computation=canonical_hash({k:v for k,v in asdict(cfg).items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')})
    if computation!=prov['config_computation_hash'] or cfg.arm!='energy' or cfg.backend!='cpu':raise IdentityError('Scoring config/provenance mismatch')
    current=code_provenance(ROOT)
    for key in ('code_fingerprint','code_files','lock_sha256','python','dependencies'):
        if prov[key]!=current[key]:raise IdentityError('Scoring numerical source/environment mismatch')
    pins={role:hashlib.sha256((run/(role+'.msgpack')).read_bytes()).hexdigest() for role in ('best','last')}
    analysis_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (output/'score.json').exists():
        score_bytes=(output/'score.json').read_bytes()
        if hashlib.sha256(score_bytes).hexdigest()!=(output/'score.sha256').read_text().strip():raise ValueError('Cached score fields changed')
        saved=json.loads(score_bytes)
        if saved['checkpoint_sha256']!=pins or saved['analysis_source_sha256']!=analysis_hash:raise ValueError('Existing score belongs to different checkpoint/code')
        for role,x in saved['prediction_files'].items():
            if hashlib.sha256((output/(role+'-validation.npz')).read_bytes()).hexdigest()!=x:raise ValueError('Cached prediction evidence changed')
        return saved
    if output.exists() and any(output.iterdir()):raise ValueError('Incomplete score directory retained; use a fresh analysis directory')
    output.mkdir(parents=True,exist_ok=True)
    bundle=load_training_bundle(cfg.data_directory,expected_split_hash=cfg.expected_split_hash)
    if prov['dataset_identity']!=bundle.manifest['identity'] or prov['normalization']!=asdict(bundle.normalization):raise ValueError('Scoring dataset/scales mismatch')
    model,predictor=model_and_predictor(cfg);initial,_=sample_training_batch(bundle,np.random.PCG64(cfg.seed).state,cfg.batch_size)
    template=initialize_state(model,initial,bundle.normalization,cfg.seed,cfg.learning_rate)
    if any(prov[k]!=v for k,v in initialization_identity(template).items()):raise ValueError('Scoring initialization mismatch')
    last=inspect_checkpoint(run/'last.msgpack',prov);hist=history_metrics(last['control']['validation_history'],cfg.max_updates)
    if last['update']!=cfg.max_updates or hist['best_update']!=last['control']['best_update']:raise ValueError('Scoring requires complete selected fit')
    summary=read_json(run/'summary.json')
    if summary['status']!='passed' or summary['stop_reason']!='max_updates' or summary['updates']!=cfg.max_updates:raise ValueError('Scoring incomplete fit')
    predict=jax.jit(lambda par,b:predictor(par,b,model=model));part=bundle.validation;groups=[part.memberships[r]['groups'][0] for r in part.record_ids]
    results={};prediction_files={}
    for role in ('best','last'):
        state=load_checkpoint(run/(role+'.msgpack'),prov,template=template);energies=[];forces=[]
        for start in range(0,len(part.record_ids),cfg.batch_size):
            batch=partition_batch(part,np.arange(start,min(start+cfg.batch_size,len(part.record_ids))))
            e,f=jax.device_get(predict(state['params'],batch));assert_finite((e,f));energies.extend(e);forces.extend(np.asarray(f).reshape(-1,part.positions.shape[1],3))
        e=np.asarray(energies);f=np.asarray(forces)
        m=prediction_metrics(list(part.record_ids),groups,e,f,part.energy_residual,part.forces)
        m['pooled_component_force_rmse_eV_per_A']=float(np.sqrt(np.mean((f-part.forces)**2)))
        expected=hist[role+'_equal_connection_force_mae_eV_per_A']
        if not np.isclose(m['equal_connection_force_mae_eV_per_A'],expected,atol=1e-10,rtol=0):raise ValueError('Independent endpoint score mismatch')
        results[role]=m
        np.savez_compressed(output/(role+'-validation.npz'),record_ids=np.asarray(part.record_ids),groups=np.asarray(groups),energy=e,forces=f)
        prediction_files[role]=hashlib.sha256((output/(role+'-validation.npz')).read_bytes()).hexdigest()
    result={'status':'independently_scored_validation_only','seed':cfg.seed,'model':cfg.model,'history':hist,'roles':results,'checkpoint_sha256':pins,
        'prediction_files':prediction_files,'analysis_source_sha256':analysis_hash,'parameter_count':prov['parameter_count'],
        'source_fingerprint':prov['code_fingerprint'],'training_revision':prov['revision'],'final_segment_elapsed_seconds':summary['elapsed_seconds'],
        'segment_timing':segment_timings(run),
        'backend':summary['backend'],'no_test_or_recovery_claim':True}
    write_json(output/'score.json',result)
    (output/'score.sha256').write_text(hashlib.sha256((output/'score.json').read_bytes()).hexdigest()+'\n')
    return result


def score_fit(run_directory,analysis_root):
    """Reuse verified scoring, or retain an interrupted attempt and score afresh."""
    root=Path(analysis_root);root.mkdir(parents=True,exist_ok=True);attempts=sorted(p for p in root.glob('[0-9]*') if p.is_dir());rejected=[]
    for path in reversed(attempts):
        if not (path/'score.json').exists():
            rejected.append({'path':str(path),'reason':'Interrupted scoring; no completed score'})
            continue
        try:return {**score_run(run_directory,path),'score_directory':str(path)}
        except IdentityError:raise
        except Exception as exc:rejected.append({'path':str(path),'reason':f'{type(exc).__name__}: {exc}'})
    output=root/f'{max([int(p.name) for p in attempts],default=0)+1:06d}'
    write_json(root/'rejected-attempts.json',rejected)
    try:return {**score_run(run_directory,output),'score_directory':str(output)}
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True);write_json(output/'failure.json',{'reason':f'{type(exc).__name__}: {exc}'})
        raise
