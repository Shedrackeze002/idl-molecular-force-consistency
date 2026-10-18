"""Mechanical twelve-slot CPU budget; no model fitting or outcome selection."""
import argparse
import math
import numpy as np
from e3x_phase_common import ARMS,RECIPES,read_json,save_json


def positive(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Nonfinite/negative cost')
    return float(value)


def conservative_update(durations):
    if len(durations)!=100:raise ValueError('Need 100 ordinary update durations')
    a=np.array([positive(x) for x in durations])
    if np.any(a<=0):raise ValueError('Ordinary update durations must be positive')
    return max(float(np.percentile(a,95)),float(a.reshape(5,20).mean(axis=1).max()))


def io_accounting(updates,model):
    if type(updates) is not int or updates<500 or updates%500:raise ValueError('Invalid update grid')
    for key in ('validation_base_bytes','validation_event_bytes','loss_base_bytes','loss_row_bytes','checkpoint_base_bytes','checkpoint_event_bytes','fixed_output_bytes','seconds_per_byte','fixed_event_seconds','reload_and_score_seconds'):positive(model[key])
    n=updates//500;k=n+1;history=k*(k+1)//2
    values={'validation':k*model['validation_base_bytes']+history*model['validation_event_bytes'],
            'losses':k*model['loss_base_bytes']+(500*n*(n+1)//2+updates)*model['loss_row_bytes'],
            'checkpoints':2*k*model['checkpoint_base_bytes']+(2*history-1+k)*model['checkpoint_event_bytes']}
    cumulative=sum(values.values())+model['fixed_output_bytes']
    retained=(model['validation_base_bytes']+k*model['validation_event_bytes']+model['loss_base_bytes']+updates*model['loss_row_bytes']+2*(model['checkpoint_base_bytes']+k*model['checkpoint_event_bytes'])+model['fixed_output_bytes'])
    return {'writes':{'validation':k,'losses':k,'last':k,'best':k},'bytes_by_format':values,
            'cumulative_bytes':cumulative,'retained_bytes':retained,
            'seconds':cumulative*model['seconds_per_byte']+4*k*model['fixed_event_seconds']}


def forecast(cells,io,updates,*,aux_seconds=0):
    expected={(w,a) for w in (32,64) for a in ARMS};by={}
    for c in cells:
        key=(c['width'],c['arm'])
        if key in by or key not in expected or c['status']!='complete':raise ValueError('Duplicate/invalid/incomplete timing')
        if len(c['kernel_update_seconds'])!=100 or len(c['validation_seconds'])!=3:raise ValueError('Incomplete timing series')
        for x in c['kernel_update_seconds']+c['validation_seconds']:positive(x)
        if c['id']!=f'{key[0]}-{key[1]}' or any(f<k for f,k in zip(c['full_update_seconds'],c['kernel_update_seconds'])):raise ValueError('Mislabeled timing or omitted kernel time')
        by[key]=c
    if set(by)!=expected:raise ValueError('Missing timing case')
    writes=io_accounting(updates,io);recipes={}
    for r,(width,_,_) in RECIPES.items():
        recipes[r]={}
        for a in ARMS:
            c=by[width,a];ordinary=conservative_update(c['full_update_seconds'])
            setup=sum(positive(c[n]) for n in ('initialization_seconds','compilation_seconds','warmup_seconds'))
            finalization=positive(io['reload_and_score_seconds'])
            score=finalization/3+max(c['validation_seconds'])
            fit=setup+updates*ordinary+(1+updates//500)*max(c['validation_seconds'])+writes['seconds']+2*finalization/3
            cost=fit+score
            recipes[r][a]={'seconds':cost,'fit_seconds':fit,'score_seconds':score,'ordinary_update_seconds':ordinary,'representative_case':c['id']}
    slots={f'energy-{r}-0':recipes[r]['energy']['seconds'] for r in RECIPES}
    slots['selected-energy-1']=max(x['energy']['seconds'] for x in recipes.values())
    for seed in (0,1):
        slots[f'selected-direct-{seed}']=max(x['direct']['seconds'] for x in recipes.values())
        for strength in (.1,1.,10.):slots[f'selected-consistency-{strength:g}-{seed}']=max(x['consistency']['seconds'] for x in recipes.values())
    return {'updates':updates,'slots':slots,'per_recipe':recipes,'io':writes,'aux_seconds':positive(aux_seconds),
            'reserved_total_seconds':1.25*(positive(aux_seconds)+sum(slots.values())),
            'reserved_baseline_seconds':1.25*sum(slots[f'energy-{r}-0'] for r in RECIPES),
            'reserve_multiplier':1.25,'cost_scope':'serial CPU forecasts; six representative cases, not measured scalar/seed invariance'}


def choose_updates(cells,io,*,aux_seconds=0,total_seconds=86400,baseline_seconds=43200):
    feasible=[]
    for u in range(50000,200001,500):
        f=forecast(cells,io,u,aux_seconds=aux_seconds)
        if f['reserved_total_seconds']<=positive(total_seconds) and f['reserved_baseline_seconds']<=positive(baseline_seconds):feasible.append(f)
    if not feasible:raise ValueError('No feasible budget >=50000; scope decision required')
    result=feasible[-1];result['envelope_seconds']=total_seconds;result['baseline_window_seconds']=baseline_seconds
    result['margins_seconds']={'overall':total_seconds-result['reserved_total_seconds'],'baseline':baseline_seconds-result['reserved_baseline_seconds']}
    if result['updates']==200000:result['binding_constraint']='ceiling'
    else:
        nxt=forecast(cells,io,result['updates']+500,aux_seconds=aux_seconds)
        result['binding_constraint']=[n for n,exceeds in [('overall',nxt['reserved_total_seconds']>total_seconds),('baseline',nxt['reserved_baseline_seconds']>baseline_seconds)] if exceeds]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('timing');p.add_argument('output');p.add_argument('--aux-seconds',type=float,required=True);a=p.parse_args()
    x=read_json(a.timing)
    if x['status']!='complete':raise ValueError('Incomplete probes')
    result=choose_updates(x['cells'],x['io_model'],aux_seconds=a.aux_seconds)
    from pathlib import Path
    if Path(a.output).exists():raise FileExistsError(a.output)
    save_json(a.output,result)

if __name__=='__main__':main()
