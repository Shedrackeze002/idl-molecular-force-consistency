"""Fresh, serial C/B/A launches; each attempt consumes its predeclared slot."""
import argparse
from dataclasses import asdict
import math
import shutil
from pathlib import Path
import subprocess
import sys
import time
from e3x_phase_common import (ROOT,RECIPES,source_pin,require_equal_pin,fixed_config,
    dataset_pin,fresh_directory,read_json,save_json,digest,label_coverage,assert_imports)


def remaining_seconds(*deadlines,clock=None):
    remaining=min(deadlines)-(clock or time.perf_counter)()
    if remaining<=0:raise TimeoutError('Shared baseline/candidate deadline exhausted')
    return remaining


def validate_record_set(records):
    if set(records)!={'A','B','C'}:raise ValueError('Three fixed candidate records required')
    first=records['A'];u=first['config']['max_updates']
    if type(u) is not int or u not in range(50000,200001,500):raise ValueError('Invalid common U')
    for name,r in records.items():
        if r['candidate']!=name or r['config']['max_updates']!=u or r['config']['seed']!=0 or r['config']['arm']!='energy':raise ValueError('Non-common scientific budget/arm/seed')
        if any(r[k]!=first[k] for k in ('budget_sha256','source_pin','dataset_pin')):raise ValueError('Non-common source/data/budget identity')
    return u


def require_storage(path,required_bytes):
    p=Path(path)
    while not p.exists():p=p.parent
    free=shutil.disk_usage(p).free
    if required_bytes<=0 or free<required_bytes:raise OSError('Insufficient fresh free storage for retained study outputs')
    return {'free_bytes':free,'required_bytes':required_bytes}


def launch_templates(timing,budget,data,output):
    pin=source_pin();require_equal_pin(pin,timing['source_pin']);require_equal_pin(dataset_pin(data),timing['dataset_pin'])
    if timing['status']!='complete' or budget['updates'] not in range(50000,200001,500):raise ValueError('Incomplete/faulty budget')
    cells={(c['width'],c['arm']):c for c in timing['cells']};records={}
    for candidate in ('C','B','A'):
        width=RECIPES[candidate][0];cost=budget['per_recipe'][candidate]['energy']['seconds']
        fit=budget['per_recipe'][candidate]['energy']['fit_seconds'];score=budget['per_recipe'][candidate]['energy']['score_seconds']
        wall=1.25*fit
        records[candidate]={'candidate':candidate,'config':fixed_config(candidate,data,Path(output)/candidate,budget['updates'],wall),
            'source_pin':pin,'dataset_pin':timing['dataset_pin'],'initial_identity':cells[width,'energy']['initial_identity'],
            'budget_sha256':digest(__import__('e3x_phase_common').encoded(budget)),'forecast_seconds':cost,
            'fit_wall_seconds':1.25*fit,'score_wall_seconds':1.25*score,'slot_wall_seconds':1.25*cost,
            'required_remaining_storage_bytes':math.ceil(1.25*12*budget['io']['retained_bytes'])+1024**3,
            'source_rule':'full scientific fingerprint and external helper pins; original run provenance retained'}
    return records


def train_worker(launch_file):
    assert_imports()
    from benchmark_e3x_budget import cpu_context
    with cpu_context():return _train_worker(launch_file)

def _train_worker(launch_file):
    from force_consistency.train import train,RunConfig,model_and_predictor,sample_training_batch,new_sampler_state,initialize_state,initialization_identity
    from force_consistency.data.prepare import load_training_bundle
    launch=read_json(launch_file);assert_imports();require_equal_pin(source_pin(),launch['source_pin'])
    require_equal_pin(dataset_pin(launch['config']['data_directory']),launch['dataset_pin'])
    if Path(launch['config']['run_directory']).exists():raise FileExistsError('Candidate directory already exists')
    cfg=RunConfig(**launch['config']);bundle=load_training_bundle(cfg.data_directory,expected_split_hash=cfg.expected_split_hash)
    model,_=model_and_predictor(cfg);batch,_=sample_training_batch(bundle,new_sampler_state(cfg.seed),cfg.batch_size)
    initial=initialize_state(model,batch,bundle.normalization,cfg.seed,cfg.learning_rate)
    if initialization_identity(initial)!=launch['initial_identity']:raise ValueError('Candidate/timing initialization mismatch')
    result=train(launch['config'])
    if result['status']!='passed':raise RuntimeError(result.get('failure_reason',result['status']))
    require_equal_pin(source_pin(),launch['source_pin'])


def launch_candidates(records_file,output,approval_file,label_file):
    assert_imports()
    from benchmark_e3x_budget import invoke
    from evaluate_e3x_adequacy import evaluate_a1
    approval=read_json(approval_file);records=read_json(records_file)
    if approval['status']!='accepted_for_candidates':raise ValueError('Filled-record reviews required')
    u=validate_record_set(records)
    require_equal_pin(source_pin(),approval['source_pin'])
    if digest(Path(records_file).read_bytes())!=approval['launch_records_sha256']:raise ValueError('Reviewed records differ')
    data=records['A']['config']['data_directory'];manifest=read_json(Path(data)/'manifest.json');label_coverage(read_json(label_file),manifest)
    capacity=require_storage(output,max(r['required_remaining_storage_bytes'] for r in records.values()))
    root=fresh_directory(output);save_json(root/'STORAGE.json',capacity);outcomes=[];baseline_deadline=time.perf_counter()+43200
    for candidate in ('C','B','A'):
        launch=records[candidate];require_equal_pin(source_pin(),launch['source_pin'])
        require_storage(root,launch['required_remaining_storage_bytes'])
        expected=fixed_config(candidate,data,root/candidate,launch['config']['max_updates'],launch['config']['wall_seconds'])
        if expected!=launch['config']:raise ValueError('Non-fixed scientific configuration/output')
        require_equal_pin(dataset_pin(data),launch['dataset_pin'])
        slot_deadline=time.perf_counter()+launch['slot_wall_seconds']
        if launch['fit_wall_seconds']+launch['score_wall_seconds']>launch['slot_wall_seconds']+1e-8:raise ValueError('Slot allocations exceed reserved forecast')
        lp=root/(candidate+'-launch.json');save_json(lp,launch)
        save_json(root/(candidate+'-attempt.json'),{'candidate':candidate,'status':'consumed','started_monotonic':time.perf_counter()})
        try:
            invoke([sys.executable,__file__,'--train-worker',str(lp)],root/(candidate+'-training.txt'),min(launch['fit_wall_seconds'],remaining_seconds(baseline_deadline,slot_deadline)))
            invoke([sys.executable,str(ROOT/'scripts/evaluate_e3x_adequacy.py'),'--run',launch['config']['run_directory'],'--launch',str(lp),'--output',str(root/(candidate+'-score'))],root/(candidate+'-score.txt'),min(launch['score_wall_seconds'],remaining_seconds(baseline_deadline,slot_deadline)))
            outcome=read_json(root/(candidate+'-score/score.json'))
        except Exception as exc:outcome={'candidate':candidate,'status':'failed','failure_reason':f'{type(exc).__name__}: {exc}'}
        outcomes.append(outcome);save_json(root/'outcomes.json',outcomes)
    decision=evaluate_a1(outcomes,records['A']['config']['max_updates'],True)
    save_json(root/'A1.json',decision);return decision


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-worker');p.add_argument('--records');p.add_argument('--output');p.add_argument('--approval');p.add_argument('--labels');a=p.parse_args()
    if a.train_worker:train_worker(a.train_worker)
    else:
        if not all((a.records,a.output,a.approval,a.labels)):raise ValueError('Explicit reviewed inputs required')
        launch_candidates(a.records,a.output,a.approval,a.labels)

if __name__=='__main__':main()
