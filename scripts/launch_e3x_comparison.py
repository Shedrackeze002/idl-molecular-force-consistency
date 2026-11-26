"""One fixed serial comparison schedule; immutable fit, score and analysis stages."""
import argparse
import math
from pathlib import Path
import sys
import time
from e3x_phase_common import (ROOT,assert_imports,dataset_pin,digest,encoded,fixed_config,
    fresh_directory,read_json,require_equal_pin,save_json,source_pin,label_coverage)
from benchmark_e3x_budget import invoke
from launch_e3x_candidates import remaining_seconds,require_storage
from diagnose_e3x_consistency import external_scripts

ORDER=(('direct',0,0.),('consistency',0,1.),('direct',1,0.),('consistency',1,1.),
       ('consistency',0,.1),('consistency',1,.1),('consistency',0,10.),('consistency',1,10.),('energy',1,0.))


def slot_id(arm,seed,strength):
    return f'consistency-{strength:g}-{seed}' if arm=='consistency' else f'{arm}-{seed}'


def comparison_records(budget,selected_record,output_root):
    if budget['updates']!=200000 or budget['reserve_multiplier']!=1.25 or selected_record['candidate']!='C':raise ValueError('Frozen budget/recipe mismatch')
    records={};root=Path(output_root).resolve()
    for arm,seed,strength in ORDER:
        cost=budget['per_recipe']['C'][arm]
        if any(not isinstance(cost[k],(int,float)) or not math.isfinite(cost[k]) or cost[k]<=0 for k in ('fit_seconds','score_seconds','seconds')):raise ValueError('Invalid cost ledger')
        if not math.isclose(cost['seconds'],cost['fit_seconds']+cost['score_seconds'],rel_tol=0,abs_tol=1e-9):raise ValueError('Invalid slot cost')
        fit=1.25*cost['fit_seconds'];score=1.25*cost['score_seconds'];cid=slot_id(arm,seed,strength)
        if arm=='energy' and fit!=selected_record['fit_wall_seconds']:raise ValueError('Energy seed1 cap differs from original C')
        records[cid]={'candidate':cid,'config':fixed_config('C',selected_record['config']['data_directory'],root/cid,200000,fit,arm=arm,strength=strength,seed=seed),
            'source_pin':selected_record['source_pin'],'dataset_pin':selected_record['dataset_pin'],
            'initial_identity':selected_record['initial_identities'][arm][str(seed)],'external_scripts':selected_record['external_scripts'],
            'budget_sha256':digest(encoded(budget)),'fit_wall_seconds':fit,'score_wall_seconds':score,'slot_wall_seconds':1.25*cost['seconds'],
            'diagnostic_wall_seconds':180. if arm!='energy' else 0.,
            'required_remaining_storage_bytes':math.ceil(1.25*12*budget['io']['retained_bytes'])+1024**3}
    for seed in (0,1):
        paired=[r['initial_identity'] for r in records.values() if r['config']['seed']==seed and r['config']['arm']!='energy']
        if any(x!=paired[0] for x in paired):raise ValueError('Direct/consistency initial states are not paired')
    if records['direct-0']['initial_identity']==records['direct-1']['initial_identity']:raise ValueError('Seed1 reused seed0 identity')
    return records


def validate_comparison_records(records,budget,selected,output):
    if records!=comparison_records(budget,selected,output):raise ValueError('Reviewed nine-slot mapping/caps/configs differ')


def run_inventory(path):
    root=Path(path)
    return {str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(root.rglob('*')) if p.is_file()}


def completed_fit(run,record):
    root=Path(run);cfg=record['config'];summary=read_json(root/'summary.json');prov=read_json(root/'provenance.json')
    if (read_json(root/'config.json')!=cfg or summary['status']!='passed' or summary['updates']!=200000
        or summary['stop_reason']!='max_updates' or summary['provenance']!=prov):raise ValueError('Fit did not complete the frozen budget')
    for key in ('code_fingerprint','code_files','lock_sha256','python','dependencies'):
        if prov[key]!=record['source_pin'][key]:raise ValueError('Completed fit source mismatch')
    if any(prov[k]!=v for k,v in record['initial_identity'].items()):raise ValueError('Completed initialization mismatch')
    for role,name,update in [('last','checkpoint-verification.json',200000),('best','best-checkpoint-verification.json',summary['best_update'])]:
        v=read_json(root/name)
        if v['status']!='passed' or v['role']!=role or v['update']!=update:raise ValueError('Completed fit reload proof missing')
    return {'updates':200000,'best_update':summary['best_update'],'inventory':run_inventory(root)}


def require_primary_score(score,record):
    # Later arm BEST0 is an observed result, not a new A1 failure/eligibility rule.
    if (score['status']!='verified' or score['candidate']!=record['candidate'] or score['source_pin']!=record['source_pin']
        or score['updates']!=200000 or score['max_updates']!=200000 or score['stop_reason']!='max_updates'
        or score['record_count']!=974 or score['connection_count']!=5 or type(score['best_update']) is not int
        or not 0<=score['best_update']<=200000 or any(score[k] is not True for k in ('source_matches','best_reload_verified','zero_force_crosscheck'))):raise ValueError('Primary comparison score is not verified')
    value=score['equal_connection_force_mae_eV_per_A']
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Invalid primary force MAE')
    return score


def execute_slot(record,root,global_deadline):
    # An expired allocation cannot consume a fit that was never invoked.
    remaining_seconds(global_deadline)
    root=Path(root);cid=record['candidate'];launch_file=root/(cid+'-launch.json');save_json(launch_file,record)
    slot_deadline=time.perf_counter()+record['slot_wall_seconds']
    state={'candidate':cid,'fit_slot_consumed':True,'fit':'running','score':'pending','diagnostic':'pending' if record['diagnostic_wall_seconds'] else 'not_applicable',
        'score_only_corrections':0,'automatic_retry':False,'elapsed_seconds':{}}
    statefile=root/(cid+'-stages.json');save_json(statefile,state);stage='fit'
    commands={'fit':[sys.executable,str(ROOT/'scripts/launch_e3x_candidates.py'),'--train-worker',str(launch_file)],
        'score':[sys.executable,str(ROOT/'scripts/evaluate_e3x_adequacy.py'),'--run',record['config']['run_directory'],'--launch',str(launch_file),'--output',str(root/(cid+'-score'))],
        'diagnostic':[sys.executable,str(ROOT/'scripts/diagnose_e3x_consistency.py'),'--run',record['config']['run_directory'],'--launch',str(launch_file),
            '--score',str(root/(cid+'-score')),'--output',str(root/(cid+'-diagnostic'))]}
    try:
        for stage in ('fit','score','diagnostic'):
            if stage=='diagnostic' and not record['diagnostic_wall_seconds']:break
            deadlines=(global_deadline,) if stage=='diagnostic' else (global_deadline,slot_deadline)
            cap=record[{'fit':'fit_wall_seconds','score':'score_wall_seconds','diagnostic':'diagnostic_wall_seconds'}[stage]]
            timeout=min(cap,remaining_seconds(*deadlines))
            state[stage]='running';save_json(statefile,state);started=time.perf_counter()
            try:invoke(commands[stage],root/(cid+'-'+stage+'.txt'),timeout)
            finally:state['elapsed_seconds'][stage]=time.perf_counter()-started
            if stage=='fit':state['fit_evidence']=completed_fit(record['config']['run_directory'],record)
            elif stage=='score':
                state['primary_score']=require_primary_score(read_json(root/(cid+'-score/score.json')),record)
            else:
                diagnostic=read_json(root/(cid+'-diagnostic/diagnostic.json'))
                if diagnostic['status']!='verified_descriptive' or diagnostic['candidate']!=cid:raise ValueError('Diagnostic output missing')
            if stage!='fit' and run_inventory(record['config']['run_directory'])!=state['fit_evidence']['inventory']:raise ValueError('Completed fit artifacts changed during analysis')
            state[stage]='completed';save_json(statefile,state)
    except Exception as exc:
        state[stage]='failed';state['failure_reason']=f'{type(exc).__name__}: {exc}'
    finally:
        if 'fit_evidence' in state:
            try:
                state['fit_artifacts_unchanged']=run_inventory(record['config']['run_directory'])==state['fit_evidence']['inventory']
                if not state['fit_artifacts_unchanged']:raise ValueError('Completed fit artifacts changed during analysis')
            except Exception as exc:
                state['fit_artifacts_unchanged']=False;state['artifact_audit_failure']=f'{type(exc).__name__}: {exc}'
                state[stage]='failed';state.setdefault('failure_reason',state['artifact_audit_failure'])
        save_json(statefile,state)
    return state


def matching_decision(scores):
    def pair(ids):
        if any(i not in scores or scores[i].get('status')!='verified' for i in ids):return None
        values=[scores[i]['equal_connection_force_mae_eV_per_A'] for i in ids]
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid matching score')
        updates=[scores[i]['best_update'] for i in ids]
        if any(type(x) is not int or not 0<=x<=200000 for x in updates):raise ValueError('Invalid matching BEST update')
        return {'seed_scores':values,'best_updates':updates,'mean':sum(values)/2,'range':max(values)-min(values)}
    direct=pair(['direct-0','direct-1']);strengths={};matches=[]
    for strength in (.1,1.,10.):
        p=pair([slot_id('consistency',seed,strength) for seed in (0,1)]);strengths[f'{strength:g}']=p
        if p is not None and direct is not None:
            gap=abs(p['mean']-direct['mean'])
            if gap<=.1*min(p['mean'],direct['mean']) and gap<=p['range'] and gap<=direct['range']:matches.append((gap,strength))
    initial=sorted(i for i,s in scores.items() if s.get('status')=='verified' and s.get('best_update')==0)
    selected=min(matches)[1] if matches else None
    chosen=['direct-0','direct-1']+[slot_id('consistency',seed,selected) for seed in (0,1)] if matches else []
    learned=bool(matches) and not any(i in initial for i in chosen)
    return {'status':('matched' if learned else 'matched_with_initial_checkpoints') if matches else 'unmatched_or_missing_pairs','direct':direct,'consistency':strengths,
        'matched_lambdas':sorted(s for _,s in matches),'selected_lambda':min(matches)[1] if matches else None,
        'initial_checkpoint_candidates':initial,'learned_comparison_claim_supported':learned,
        'initial_checkpoint_disclosure':'BEST update 0 selects an unlearned initial checkpoint; numerical matching involving it does not establish an informative learned-force comparison',
        'selection_basis':'Two-seed BEST force MAE only; no diagnostic/energy/recovery/test selection',
        'fallback':'Report error-cost-recovery trade-off if unmatched; no extra configurations'}


def run_comparison(records_file,output,approval_file):
    assert_imports()
    envelope=read_json(records_file);approval=read_json(approval_file)
    if approval['status']!='accepted_for_comparison' or digest(Path(records_file).read_bytes())!=approval['records_sha256']:raise ValueError('Reviewed filled records required')
    reviews=approval['review_sha256']
    if len(reviews)!=2 or len({str(Path(p).resolve()) for p in reviews})!=2 or len(set(reviews.values()))!=2:raise ValueError('Two independent filled-record reviews required')
    for name,h in approval['review_sha256'].items():
        if digest(Path(name).read_bytes())!=h:raise ValueError('Review binding differs')
    records=envelope['records'];selected=envelope['selected_record'];budget=envelope['budget']
    validate_comparison_records(records,budget,selected,output)
    require_equal_pin(source_pin(),selected['source_pin'])
    if external_scripts()!=selected['external_scripts'] or external_scripts()!=approval['external_scripts']:raise ValueError('External helper pins differ')
    for path,h in envelope['frozen_inputs'].items():
        if digest(Path(path).read_bytes())!=h:raise ValueError('Frozen input changed')
    data=selected['config']['data_directory'];require_equal_pin(dataset_pin(data),selected['dataset_pin'])
    label_coverage(read_json(envelope['label_report']),read_json(Path(data)/'manifest.json'))
    if any(Path(r['config']['run_directory']).exists() for r in records.values()):raise FileExistsError('A training output already exists')
    require_storage(output,max(r['required_remaining_storage_bytes'] for r in records.values()))
    allocation=sum(r['slot_wall_seconds']+1.25*r['diagnostic_wall_seconds'] for r in records.values())
    margin=envelope['phase_reconciliation']['margin_seconds']
    if envelope['global_wall_seconds']!=allocation or isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<0:raise ValueError('Global allocation/reconciliation differs')
    root=fresh_directory(output);save_json(root/'SESSION.json',{'records_sha256':approval['records_sha256'],'external_scripts':external_scripts(),'order':[slot_id(*x) for x in ORDER]})
    deadline=time.perf_counter()+allocation;outcomes=[]
    for arm,seed,strength in ORDER:
        cid=slot_id(arm,seed,strength);record=records[cid]
        require_equal_pin(source_pin(),record['source_pin'])
        if external_scripts()!=record['external_scripts']:raise ValueError('External helper changed before slot')
        require_equal_pin(dataset_pin(data),record['dataset_pin']);require_storage(root,record['required_remaining_storage_bytes'])
        for path,h in envelope['frozen_inputs'].items():
            if digest(Path(path).read_bytes())!=h:raise ValueError('Frozen input changed before slot')
        state=execute_slot(record,root,deadline);outcomes.append(state);save_json(root/'outcomes.json',outcomes)
        if any(state[k]=='failed' for k in ('fit','score','diagnostic')):
            save_json(root/'STOPPED.json',{'candidate':cid,'state':state,'reason':'No automatic retry; completed fit preserved; review correction before continuing'})
            raise RuntimeError('Comparison stopped at '+cid+': '+state['failure_reason'])
    decision=matching_decision({s['candidate']:s['primary_score'] for s in outcomes});save_json(root/'MATCHING.json',decision)
    save_json(root/'COMPLETE.json',{'status':'all_nine_scored_result_reviews_pending','records_sha256':approval['records_sha256'],'matching':decision,'A1_redecided':False})
    return decision


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True);p.add_argument('--approval',required=True);a=p.parse_args()
    run_comparison(a.records,a.output,a.approval)

if __name__=='__main__':main()
