"""Fresh-process exhaustive BEST validation scoring and once-only v3 A1 decision."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
from e3x_phase_common import (SPLIT,read_json,save_json,source_pin,require_equal_pin,
    assert_imports,dataset_pin,fresh_directory,digest,label_coverage)


def prediction_metrics(ids,groups,energy,forces,reference_energy,reference_forces):
    e,f,re,rf=map(np.asarray,(energy,forces,reference_energy,reference_forces))
    n=len(ids)
    if n==0 or len(set(ids))!=n or len(groups)!=n or e.shape!=re.shape or e.shape!=(n,) or f.shape!=rf.shape or f.ndim!=3 or f.shape[0]!=n or f.shape[-1]!=3:raise ValueError('Invalid/duplicate prediction records or shapes')
    if not all(np.isfinite(a).all() for a in (e,f,re,rf)):raise ValueError('Nonfinite predictions')
    per=np.abs(f-rf).mean(axis=(1,2));zero=np.abs(rf).mean(axis=(1,2))
    groups=np.asarray(groups);unique=sorted(set(groups));means={g:float(per[groups==g].mean()) for g in unique}
    return {'record_count':n,'connection_count':len(unique),'per_connection_force_mae_eV_per_A':means,
            'equal_connection_force_mae_eV_per_A':float(np.mean(list(means.values()))),
            'pooled_force_mae_eV_per_A':float(np.abs(f-rf).mean()),'energy_mae_eV':float(np.abs(e-re).mean()),
            'zero_force_equal_connection_mae_eV_per_A':float(np.mean([zero[groups==g].mean() for g in unique]))}


def eligible(r,updates):
    score=r.get('equal_connection_force_mae_eV_per_A')
    return (r.get('status')=='verified' and type(r.get('best_update')) is int and 0<r['best_update']<=updates
            and r.get('updates')==updates and r.get('max_updates')==updates and r.get('stop_reason')=='max_updates'
            and r.get('record_count')==974 and r.get('connection_count')==5 and r.get('source_matches') is True
            and r.get('best_reload_verified') is True and r.get('zero_force_crosscheck') is True
            and isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and score>=0)


def evaluate_a1(records,updates,labels_verified):
    if not labels_verified:raise ValueError('Label coverage prerequisite missing')
    if len(records)!=3 or {r['candidate'] for r in records}!={'A','B','C'}:raise ValueError('All three unique consumed candidate slots required')
    good=[r for r in records if eligible(r,updates)]
    selected=min(good,key=lambda x:(x['equal_connection_force_mae_eV_per_A'],x['candidate'])) if good else None
    return {'a1_passed':selected is not None and selected['equal_connection_force_mae_eV_per_A']<=.05,
            'selected_candidate':selected['candidate'] if selected else None,'selected_score':selected['equal_connection_force_mae_eV_per_A'] if selected else None,
            'threshold_eV_per_A':.05,'consumed_slots':3,'eligible_candidates':[r['candidate'] for r in good],
            'disclosure':'Selection and adequacy use the same five validation connections; optimistic floor test. No test or recovery claim.'}


def verify_selection_history(history,updates,interval,selected_update,selected_score):
    expected=list(range(0,updates+1,interval))
    if [h['update'] for h in history]!=expected:raise ValueError('Incomplete/duplicate validation schedule')
    scores=[h['equal_connection_force_mae_eV_per_A'] for h in history]
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in scores):raise ValueError('Invalid validation score history')
    first=min(range(len(scores)),key=scores.__getitem__)
    if history[first]['update']!=selected_update or not np.isclose(scores[first],selected_score,rtol=1e-10,atol=1e-10):raise ValueError('BEST is not first validation minimum')
    return history[first]



def verify_embedded_best(checkpoint,provenance):
    """Compare authenticated original bytes; template restore can reorder dictionaries."""
    from force_consistency.runs import _read_checkpoint,load_run_control
    envelope=_read_checkpoint(checkpoint,provenance)
    control=load_run_control(checkpoint,provenance)
    if envelope['payload']!=control['best_state']:raise ValueError('BEST embedded state mismatch')
    return {'payload_sha256':digest(envelope['payload']),
            'control_sha256':envelope['control_sha256'],
            'embedded_best_state_sha256':digest(control['best_state']),
            'exact_original_payload_match':True}


def score_best(run,launch,output):
    from benchmark_e3x_budget import cpu_context
    with cpu_context():return _score_best(run,launch,output)


def _score_best(run,launch,output):
    import jax
    from force_consistency.train import RunConfig,model_and_predictor,initialize_state,initialization_identity
    from force_consistency.runs import load_checkpoint,load_run_control,assert_finite
    from force_consistency.metrics import partition_batch
    from force_consistency.data.prepare import load_training_bundle
    assert_imports();require_equal_pin(source_pin(),launch['source_pin'])
    run=Path(run);cfg_dict=read_json(run/'config.json');cfg=RunConfig(**cfg_dict)
    if asdict(cfg)!=launch['config']:raise ValueError('Run/launch configuration mismatch')
    require_equal_pin(dataset_pin(cfg.data_directory),launch['dataset_pin'])
    summary=read_json(run/'summary.json');prov=read_json(run/'provenance.json')
    if prov!=summary['provenance']:raise ValueError('Summary provenance mismatch')
    from force_consistency.data.splits import canonical_hash
    computation=canonical_hash({k:v for k,v in asdict(cfg).items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')})
    if prov['config_computation_hash']!=computation or prov['arm']!=cfg.arm or prov['loss_weights']!=cfg.loss_weights or prov['consistency_strength']!=cfg.consistency_strength:raise ValueError('Objective/computation mismatch')
    for key in ('code_fingerprint','code_files','lock_sha256','python','dependencies'):
        if prov[key]!=launch['source_pin'][key]:raise ValueError('Scientific source mismatch')
    bundle=load_training_bundle(cfg.data_directory,expected_split_hash=SPLIT)
    if prov['dataset_identity']!=bundle.manifest['identity'] or prov['normalization']!=asdict(bundle.normalization):raise ValueError('Dataset/normalization mismatch')
    # The validated loader authenticates policy/unit hashes and all permitted physical arrays.
    if any(prov[key]!=bundle.manifest[key] for key in ('split_hash','policy_hash','unit_evidence_hash')):raise ValueError('Policy/unit mismatch')
    model,predictor=model_and_predictor(cfg);b=partition_batch(bundle.train,np.arange(4))
    template=initialize_state(model,b,bundle.normalization,cfg.seed,cfg.learning_rate)
    identity=initialization_identity(template)
    if any(identity[k]!=prov[k] or identity[k]!=launch['initial_identity'][k] for k in identity):raise ValueError('Initial state identity mismatch')
    state=load_checkpoint(run/'best.msgpack',prov,template=template)
    control=load_run_control(run/'best.msgpack',prov)
    embedded_identity=verify_embedded_best(run/'best.msgpack',prov)
    verify=read_json(run/'best-checkpoint-verification.json')
    if verify['status']!='passed' or verify['role']!='best' or verify['update']!=state['update']:raise ValueError('BEST fresh-process reload missing')
    if summary['updates']!=cfg.max_updates or summary['stop_reason']!='max_updates':raise ValueError('Incomplete training budget')
    last=load_checkpoint(run/'last.msgpack',prov,template=template)
    last_verify=read_json(run/'checkpoint-verification.json')
    if int(last['update'])!=cfg.max_updates or last_verify['status']!='passed' or last_verify['role']!='last' or last_verify['update']!=cfg.max_updates:raise ValueError('Full-budget LAST state/reload missing')
    history=read_json(run/'validation.json')
    selected_event=verify_selection_history(history,cfg.max_updates,cfg.validation_interval,int(state['update']),summary['best_equal_connection_force_mae_eV_per_A'])
    expected_ids=list(bundle.validation.record_ids);expected_groups=set(bundle.manifest['splits']['groups']['aspirin']['validation'])
    if any(h['evaluated_ids']!=expected_ids or h['record_count']!=len(expected_ids) or set(h['per_group'])!=expected_groups for h in history):raise ValueError('Validation history support mismatch')
    best_event=[selected_event]
    if len(best_event)!=1 or summary['best_update']!=int(state['update']) or control['best_update']!=int(state['update']):raise ValueError('BEST selection-state mismatch')
    predict=jax.jit(lambda p,b:predictor(p,b,model=model));part=bundle.validation;energies=[];forces=[]
    for start in range(0,len(part.record_ids),cfg.batch_size):
        batch=partition_batch(part,np.arange(start,min(start+cfg.batch_size,len(part.record_ids))))
        e,f=jax.device_get(predict(state['params'],batch));assert_finite((e,f));energies.extend(np.asarray(e));forces.extend(np.asarray(f).reshape(-1,part.positions.shape[1],3))
    ids=list(part.record_ids);groups=[]
    for rid in ids:
        membership=part.memberships[rid]
        if membership['role']=='minimum' or len(membership['groups'])!=1:raise ValueError('Invalid validation membership')
        groups.append(membership['groups'][0])
    e=np.asarray(energies);f=np.asarray(forces);metrics=prediction_metrics(ids,groups,e,f,part.energy_residual,part.forces)
    if set(groups)!=set(bundle.manifest['splits']['groups']['aspirin']['validation']):raise ValueError('Wrong validation connections')
    for key in ('pooled_force_mae_eV_per_A','equal_connection_force_mae_eV_per_A','energy_mae_eV'):
        if not np.isclose(metrics[key],best_event[0][key],rtol=1e-10,atol=1e-10):raise ValueError('Independent BEST metric mismatch: '+key)
    if not np.isclose(control['best_score'],metrics['equal_connection_force_mae_eV_per_A'],rtol=1e-10,atol=1e-10) or not np.isclose(summary['best_equal_connection_force_mae_eV_per_A'],metrics['equal_connection_force_mae_eV_per_A'],rtol=1e-10,atol=1e-10):raise ValueError('Selected score mismatch')
    zero=metrics['zero_force_equal_connection_mae_eV_per_A']
    if history[0]['update']!=0 or not np.isclose(zero,history[0]['equal_connection_force_mae_eV_per_A'],rtol=1e-12,atol=1e-12):raise ValueError('Zero-force/update-0 mismatch')
    out=fresh_directory(output)
    np.savez_compressed(out/'best-validation.npz',record_ids=np.asarray(ids),groups=np.asarray(groups),energy=e,forces=f,
                        reference_energy=part.energy_residual,reference_forces=part.forces)
    result={**metrics,'candidate':launch['candidate'],'status':'verified','best_update':int(state['update']),
        'updates':summary['updates'],'max_updates':cfg.max_updates,'stop_reason':summary['stop_reason'],
        'actual_backend':'cpu','source_matches':True,'best_reload_verified':True,'zero_force_crosscheck':True,'source_pin':launch['source_pin'],
        'best_checkpoint_sha256':digest((run/'best.msgpack').read_bytes()),'prediction_sha256':digest((out/'best-validation.npz').read_bytes()),
        'embedded_best_identity':embedded_identity}
    save_json(out/'score.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--launch',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    score_best(a.run,read_json(a.launch),a.output)

if __name__=='__main__':main()
