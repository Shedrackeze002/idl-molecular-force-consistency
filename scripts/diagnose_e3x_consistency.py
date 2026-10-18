"""Descriptive BEST direct/energy-head agreement; never a selection criterion."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from e3x_phase_common import (ROOT,SPLIT,assert_imports,dataset_pin,digest,encoded,fresh_directory,
    read_json,require_equal_pin,save_json,source_pin)


def external_scripts():
    return {n:digest((ROOT/'scripts'/n).read_bytes()) for n in
            ('launch_e3x_comparison.py','diagnose_e3x_consistency.py')}


def diagnostic_metrics(groups,direct_forces,energy_head_forces,reference_forces,force_scale):
    f,fe,ref=map(np.asarray,(direct_forces,energy_head_forces,reference_forces))
    g=np.asarray(groups)
    if (isinstance(force_scale,bool) or not isinstance(force_scale,(int,float))
        or not math.isfinite(force_scale) or force_scale<=0):raise ValueError('Invalid force scale')
    if (f.ndim!=3 or f.shape[-1]!=3 or 0 in f.shape or f.shape!=fe.shape or f.shape!=ref.shape
        or g.shape!=(len(f),) or not all(np.isfinite(a).all() for a in (f,fe,ref))):raise ValueError('Invalid diagnostic alignment/shapes/values')
    unique=sorted(set(g.tolist()))
    if any(not isinstance(x,str) or not x for x in unique):raise ValueError('Invalid diagnostic groups')
    residual=f-fe;mse=float(np.mean(residual**2))
    return {'record_count':len(f),'connection_count':len(unique),
        'pooled_energy_head_force_mae_eV_per_A':float(np.mean(np.abs(fe-ref))),
        'equal_connection_energy_head_force_mae_eV_per_A':float(np.mean([np.mean(np.abs(fe[g==x]-ref[g==x])) for x in unique])),
        'residual_mse_eV2_per_A2':mse,'residual_rmse_eV_per_A':math.sqrt(mse),
        'pooled_residual_mae_eV_per_A':float(np.mean(np.abs(residual))),
        'equal_connection_residual_mae_eV_per_A':float(np.mean([np.mean(np.abs(residual[g==x])) for x in unique])),
        'normalized_residual_mse':mse/force_scale**2}


def diagnose_best(run,launch_record,verified_score_directory,output):
    assert_imports()
    if external_scripts()!=launch_record['external_scripts']:raise ValueError('External script pin mismatch')
    require_equal_pin(source_pin(),launch_record['source_pin'])
    from benchmark_e3x_budget import cpu_context
    with cpu_context():return _diagnose_best(Path(run),launch_record,Path(verified_score_directory),output)


def _diagnose_best(run,launch,score_root,output):
    import jax
    from force_consistency.train import RunConfig,model_and_predictor,initialize_state,initialization_identity
    from force_consistency.runs import load_checkpoint,assert_finite
    from force_consistency.metrics import partition_batch
    from force_consistency.objectives import energy_derived_forces
    from force_consistency.data.prepare import load_training_bundle
    from evaluate_e3x_adequacy import verify_embedded_best,prediction_metrics
    cfg=RunConfig(**read_json(run/'config.json'))
    if asdict(cfg)!=launch['config'] or cfg.arm not in ('direct','consistency'):raise ValueError('Diagnostic run/config/arm mismatch')
    require_equal_pin(dataset_pin(cfg.data_directory),launch['dataset_pin'])
    prov=read_json(run/'provenance.json');score=read_json(score_root/'score.json')
    predfile=score_root/'best-validation.npz';checkpoint=run/'best.msgpack'
    if (score['status']!='verified' or score['candidate']!=launch['candidate'] or score['source_pin']!=launch['source_pin']
        or score['best_checkpoint_sha256']!=digest(checkpoint.read_bytes())
        or score['prediction_sha256']!=digest(predfile.read_bytes())):raise ValueError('Primary score/checkpoint identity mismatch')
    identity=verify_embedded_best(checkpoint,prov)
    bundle=load_training_bundle(cfg.data_directory,expected_split_hash=SPLIT);part=bundle.validation
    expected_ids=list(part.record_ids);expected_groups=[part.memberships[r]['groups'][0] for r in expected_ids]
    with np.load(predfile,allow_pickle=False) as z:
        ids=z['record_ids'].tolist();groups=z['groups'].tolist();direct=z['forces'].copy();ref=z['reference_forces'].copy()
        if ids!=expected_ids or len(set(ids))!=len(ids) or groups!=expected_groups:raise ValueError('Primary record/group mismatch')
        if not np.array_equal(ref,part.forces) or not np.array_equal(z['reference_energy'],part.energy_residual):raise ValueError('Primary reference mismatch')
        reduced=prediction_metrics(ids,groups,z['energy'],direct,z['reference_energy'],ref)
    for k,v in reduced.items():
        if isinstance(v,(int,float)) and not np.isclose(v,score[k],rtol=1e-10,atol=1e-10):raise ValueError('Primary score reduction mismatch')
    model,predictor=model_and_predictor(cfg)
    template=initialize_state(model,partition_batch(bundle.train,np.arange(4)),bundle.normalization,cfg.seed,cfg.learning_rate)
    initial=initialization_identity(template)
    if initial!=launch['initial_identity'] or any(prov[k]!=v for k,v in initial.items()):raise ValueError('Initial state mismatch')
    state=load_checkpoint(checkpoint,prov,template=template)
    if int(state['update'])!=score['best_update']:raise ValueError('Selected update mismatch')
    predict=jax.jit(lambda b:predictor(state['params'],b,model=model))
    gradient=jax.jit(lambda b:energy_derived_forces(state['params'],b,model=model))
    implied=[]
    for start in range(0,len(ids),cfg.batch_size):
        end=min(start+cfg.batch_size,len(ids));batch=partition_batch(part,np.arange(start,end))
        _,f=jax.device_get(predict(batch));fe=jax.device_get(gradient(batch));assert_finite((f,fe))
        f=np.asarray(f).reshape(-1,part.positions.shape[1],3);fe=np.asarray(fe).reshape(f.shape)
        if not np.allclose(f,direct[start:end],rtol=1e-10,atol=1e-10):raise ValueError('Diagnostic force supplier differs from primary score')
        implied.extend(fe)
    implied=np.asarray(implied);metrics=diagnostic_metrics(groups,direct,implied,ref,bundle.normalization.force_loss_scale_eV_per_A)
    require_equal_pin(source_pin(),launch['source_pin'])
    if external_scripts()!=launch['external_scripts']:raise ValueError('External script changed')
    out=fresh_directory(output)
    np.savez_compressed(out/'consistency-validation.npz',record_ids=np.asarray(ids),groups=np.asarray(groups),
        direct_forces=direct,energy_head_forces=implied,residual=direct-implied,reference_forces=ref)
    result={**metrics,'status':'verified_descriptive','candidate':launch['candidate'],'arm':cfg.arm,'seed':cfg.seed,
        'lambda':cfg.consistency_strength,'best_update':int(state['update']),'primary_score_sha256':digest((score_root/'score.json').read_bytes()),
        'primary_prediction_sha256':score['prediction_sha256'],'best_checkpoint_sha256':score['best_checkpoint_sha256'],
        'diagnostic_prediction_sha256':digest((out/'consistency-validation.npz').read_bytes()),'source_pin':launch['source_pin'],
        'dataset_pin':launch['dataset_pin'],'launch_record_sha256':digest(encoded(launch)),
        'external_scripts':launch['external_scripts'],'embedded_best_identity':identity,
        'force_scale_eV_per_A':bundle.normalization.force_loss_scale_eV_per_A,
        'selection_use':'none; no eligibility threshold; finite-record agreement is not global conservativity'}
    save_json(out/'diagnostic.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--launch',required=True)
    p.add_argument('--score',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    diagnose_best(a.run,read_json(a.launch),a.score,a.output)

if __name__=='__main__':main()
