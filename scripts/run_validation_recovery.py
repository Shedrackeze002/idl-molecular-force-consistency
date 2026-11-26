"""Fixed validation-only controls, A2 and matched recovery; no model selection."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time

import numpy as np
from e3x_phase_common import (ROOT,assert_imports,read_json,save_json,encoded,digest,
                              source_pin,require_equal_pin,dataset_pin,fresh_directory,fixed_config)
# The existing supervisor pins src/scripts. Evaluation lives at repository root.
sys.path.insert(0,str(ROOT))
from benchmark_e3x_budget import invoke
from launch_e3x_candidates import remaining_seconds,require_storage
from evaluation.adapter import _validated_positions
from evaluation.runner import (evaluate_starts,_own_reference,_bind_split_manifest,
    _manifest_synthetic_declaration,_validate_connection_partitions,_connection_partition)
from evaluation.relax import _check_start,_validate_procedure
from evaluation.reporter import build_report,render,_plain_json_field,freeze_start_set_hash
from evaluation.validation_controls import validation_inputs,geometry_controls

MODEL_ORDER=('energy-0','direct-0','consistency-0.1-0','direct-1','consistency-0.1-1')
GROUPS=('13','18','18_2','66','78')
EXPECTED_KEYS={f'aspirin/{g}/{b}' for g in GROUPS for b in ('minus','plus')}


def safe_json(value):
    """Preserve explicit nonfinite failure facts in standards-compliant JSON."""
    bad=[]
    def clean(x,path):
        if isinstance(x,float) and not math.isfinite(x):
            bad.append({'path':path,'value':str(x)});return None
        if isinstance(x,dict):return {k:clean(v,path+'/'+k) for k,v in x.items()}
        if isinstance(x,list):return [clean(v,path+'/'+str(i)) for i,v in enumerate(x)]
        return x
    result=clean(_plain_json_field(value),'')
    if bad:result['_nonfinite_values']=bad
    return result


def atomic_json(path,value):
    from force_consistency.runs import atomic_write
    path=Path(path)
    if path.exists():raise FileExistsError('Refuse repeated immutable result '+str(path))
    atomic_write(path,encoded(safe_json(value)))


def run_model_pass(starts,arm,procedure,classification,references,policy,manifest,*,seen_minimum,output):
    """Prevalidate every start, then persist each branch before the next call."""
    _bind_split_manifest(manifest,synthetic_declaration=_manifest_synthetic_declaration(manifest))
    _validate_connection_partitions(starts,manifest);_validate_procedure(procedure)
    for key,entry in sorted(starts.items()):
        if _connection_partition(key,manifest)!='validation':raise ValueError('Only validation branches admitted')
        _own_reference(key,classification,references)
        array=_validated_positions(entry['positions'],arm['species'],entry['species'],arm['arm_id'])
        _check_start(array[None] if array.ndim==2 else array)
    root=fresh_directory(output);raw=root/'raw-branches';raw.mkdir();branches={};started=time.perf_counter();key=None
    try:
        for index,(key,entry) in enumerate(sorted(starts.items())):
            result=evaluate_starts({key:entry},arm,procedure,classification,references,policy,
                                   split_manifest=manifest,seen_minimum=seen_minimum,authorise_test_partition=False)
            row=result['branches'][key]
            if 'checkpoint_identity' in arm:row['_audit_record']['arm']['checkpoint_identity']=arm['checkpoint_identity']
            atomic_json(raw/f'{index:03d}.json',row);branches[key]=row
        report,diagnostics=build_report(branches)
        report=safe_json(report);diagnostics=safe_json(diagnostics)
        render(report,diagnostics,root,'recovery')
        payload={'complete':True,'branches':_plain_json_field(branches),'report':report,'diagnostics':diagnostics,
                 'elapsed_seconds':time.perf_counter()-started,'force_evaluation_count':sum(r['force_evaluation_count'] for r in branches.values())}
        atomic_json(root/'result.json',payload);return payload
    except Exception as exc:
        atomic_json(root/'INCOMPLETE.json',{'complete':False,'completed_branch_keys':list(branches),'failed_branch':key,
            'failure_reason':f'{type(exc).__name__}: {exc}','elapsed_seconds':time.perf_counter()-started,
            'partial_rows_are_operational_evidence_not_complete_report':True})
        raise


def _validated_result(payload):
    if not payload or payload.get('complete') is not True:return None
    rows=payload.get('branches',{})
    if len(rows)!=10:return None
    if set(rows)!=EXPECTED_KEYS or any(r.get('partition')!='validation' for r in rows.values()):raise ValueError('Unexpected validation branch support')
    build_report(rows)
    connection={g:sum(rows[f'aspirin/{g}/{b}']['category']=='matched_reference' for b in ('minus','plus'))/2 for g in GROUPS}
    recovered=sum(r['category']=='matched_reference' for r in rows.values())
    return {'status':'complete','recovered_branches':recovered,'branch_denominator':10,'connection_rates':connection,'rate':recovered/10}


def _context(recovered,control):
    if control.get('nearest_context_status')!='verified':return 'nearest-reference context unavailable'
    count=control['nearest_own_class_count']
    if type(count) is not int or not 0<=count<=10:raise ValueError('Invalid nearest context count')
    return ('relaxation adds no evidence beyond geometric proximity at this accuracy' if recovered<=count
            else 'recovery exceeds nearest-reference context count')


def evaluate_a2(baseline_report,control):
    result=_validated_result(baseline_report)
    if result is None or control.get('own_geometry_status')!='verified':
        return {'status':'withheld','reason':'Incomplete baseline or unverified measured no-relaxation control'}
    count=control['own_geometry_count']
    if type(count) is not int or not 0<=count<=10:raise ValueError('Invalid no-relaxation count')
    recovered=result['recovered_branches']
    return {'status':'passed' if recovered>=1 and recovered>count else 'failed',
            'recovered_branches':recovered,'no_relaxation_count':count,'branch_denominator':10,
            'nearest_reference_context':_context(recovered,control),
            'limitation':'Ten paired branches from five connections; A2 failure is weak evidence; no tuning/retry'}


def recovery_summary(reports,control,a2):
    models={cid:_validated_result(reports.get(cid)) for cid in MODEL_ORDER};arms={}
    for arm in ('direct','consistency-0.1'):
        pair=[models[f'{arm}-{seed}'] for seed in (0,1)]
        if any(x is None for x in pair):arms[arm]={'status':'withheld'}
        else:
            rates=[x['rate'] for x in pair];arms[arm]={'status':'complete','seed_rates':rates,
                'mean_rate':sum(rates)/2,'seed_range':max(rates)-min(rates)}
    if all(x is not None for x in models.values()):
        differences={g:sum(models[f'consistency-0.1-{s}']['connection_rates'][g]-models[f'direct-{s}']['connection_rates'][g] for s in (0,1))/2 for g in GROUPS}
        status='complete_validation_comparison' if a2['status']=='passed' else 'complete_descriptive_A2_'+a2['status']
    else:differences=None;status='withheld_incomplete_models'
    for item in models.values():
        if item is not None:item['nearest_reference_context']=_context(item['recovered_branches'],control)
    return {'status':status,'models':models,'arms':arms,'A2':a2,'paired_connection_differences':differences,
        'mean_paired_difference':sum(differences.values())/5 if differences is not None else None,
        'independent_connection_count':5,'seeds_per_arm':2,'scientific_main_hypothesis_established':False,
        'scope':'Paired validation operational recovery; direct fixed point is not automatically an energy minimum; no main-test claim'}


def evaluation_pin():
    files=list((ROOT/'evaluation').glob('*.py'))+[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(files)}


def recovery_records(acceptances,config,output):
    """Resolve the accepted C and exactly four accuracy-matched BEST identities."""
    controlled=read_json(acceptances['controlled']);energy=read_json(acceptances['energy'])
    if controlled['status']!='ACCEPTED_TWO_SEED_VALIDATION_COMPARISON' or controlled['selected_consistency_lambda']!=.1:raise ValueError('Accepted lambda0.1 comparison required')
    if energy['status']!='ACCEPTED_SINGLE_SEED_VALIDATION_A1_PASS' or energy['selected_candidate']!='C':raise ValueError('Accepted original C required')
    frozen=read_json(ROOT/'configs/aspirin-shape-recovery.json')
    if config['evaluation']!=frozen:raise ValueError('Frozen recovery configuration differs')
    models={};sources=source_pin();files={}
    for path in list(acceptances.values())+[config[k] for k in ('archive','manifest','classification')]:
        files[str(Path(path).resolve())]=digest(Path(path).read_bytes())
    for cid in MODEL_ORDER:
        paths=config['models'][cid];launch=read_json(paths['launch']);score=read_json(Path(paths['score_directory'])/'score.json')
        if launch['source_pin']!=sources:raise ValueError('Scientific source changed')
        cfg=launch['config'];expected_arm='energy' if cid=='energy-0' else ('direct' if cid.startswith('direct') else 'consistency')
        if cfg['arm']!=expected_arm or cfg['seed']!=int(cid[-1]) or cfg['max_updates']!=200000 or cfg['consistency_strength']!=(.1 if expected_arm=='consistency' else 0.):raise ValueError('Wrong frozen comparison model')
        expected=fixed_config('C',cfg['data_directory'],cfg['run_directory'],200000,cfg['wall_seconds'],
            arm=expected_arm,strength=.1 if expected_arm=='consistency' else 0.,seed=int(cid[-1]))
        if cfg!=expected:raise ValueError('Frozen C training recipe differs')
        if cid=='energy-0':
            checkpoint_hash=energy['selected_checkpoint_sha256'];best_update=energy['selected_best_update'] if 'selected_best_update' in energy else energy['selected_update']
        else:
            chosen=controlled['matched_pair_checkpoint_identities'][cid];checkpoint_hash=chosen['checkpoint_sha256'];best_update=chosen['best_update']
            if str(Path(paths['run_directory'])/'best.msgpack')!=chosen['checkpoint']:raise ValueError('Matched checkpoint path changed')
            if digest((Path(paths['score_directory'])/'score.json').read_bytes())!=chosen['score_sha256']:raise ValueError('Accepted matched score changed')
        if score['best_checkpoint_sha256']!=checkpoint_hash or score['best_update']!=best_update or score['status']!='verified':raise ValueError('Frozen BEST score mismatch')
        score_path=Path(paths['score_directory'])/'score.json';files[str(score_path.resolve())]=digest(score_path.read_bytes());files[str(Path(paths['launch']).resolve())]=digest(Path(paths['launch']).read_bytes())
        models[cid]={'run_directory':str(Path(paths['run_directory']).resolve()),'launch_record':launch,'score_directory':str(Path(paths['score_directory']).resolve()),
            'binding':{'checkpoint_sha256':checkpoint_hash,'best_update':best_update,'score_sha256':digest(score_path.read_bytes())},'preflight_wall_seconds':120.,'pass_wall_seconds':1800.}
    return {'order':list(MODEL_ORDER),'models':models,'evaluation':frozen,'source_pin':sources,'evaluation_pin':evaluation_pin(),
        'inputs':files,'archive':str(Path(config['archive']).resolve()),'manifest':str(Path(config['manifest']).resolve()),'classification':str(Path(config['classification']).resolve()),
        'data_directory':models['energy-0']['launch_record']['config']['data_directory'],'output':str(Path(output).resolve()),
        'control_wall_seconds':300.,'global_wall_seconds':12375.,'automatic_retry':False,'required_free_storage_bytes':1024**3}


def _verify_records(record):
    if record['order']!=list(MODEL_ORDER) or set(record['models'])!=set(MODEL_ORDER):raise ValueError('Fixed five-model order required')
    if record['evaluation']!=read_json(ROOT/'configs/aspirin-shape-recovery.json'):raise ValueError('Frozen evaluation policy changed')
    require_equal_pin(source_pin(),record['source_pin'])
    if evaluation_pin()!=record['evaluation_pin']:raise ValueError('Evaluator/source pin mismatch')
    for path,h in record['inputs'].items():
        if digest(Path(path).read_bytes())!=h:raise ValueError('Frozen input changed: '+path)
    for cid,item in record['models'].items():
        if item['preflight_wall_seconds']!=120. or item['pass_wall_seconds']!=1800.:raise ValueError('Frozen stage caps differ')
        require_equal_pin(dataset_pin(item['launch_record']['config']['data_directory']),item['launch_record']['dataset_pin'])
        if digest((Path(item['run_directory'])/'best.msgpack').read_bytes())!=item['binding']['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    if record['control_wall_seconds']!=300. or record['global_wall_seconds']!=12375. or record['automatic_retry'] is not False:raise ValueError('Frozen resource policy differs')


def _inputs(record):
    manifest=read_json(record['manifest']);classification=read_json(record['classification'])
    starts,refs,policy,binding=validation_inputs(record['archive'],manifest,classification)
    if binding!=record['start_binding']:raise ValueError('Frozen validation starts/member binding differs')
    prepared=read_json(Path(record['data_directory'])/'manifest.json');train_ids=set(prepared['partitions']['train']['record_ids'])
    seen={key:classification[key]['reference_class'] in train_ids for key in starts}
    return starts,refs,policy,binding,manifest,classification,seen


def adapter_replay(arm,score_directory,data_directory,split_hash):
    from force_consistency.data.prepare import load_training_bundle
    bundle=load_training_bundle(data_directory,expected_split_hash=split_hash);part=bundle.validation
    with np.load(Path(score_directory)/'best-validation.npz',allow_pickle=False) as z:
        if z['record_ids'].tolist()!=list(part.record_ids) or not np.array_equal(z['reference_forces'],part.forces):raise ValueError('Replay validation references differ')
        expected=z['forces'].copy()
    if len(part.record_ids)!=974:raise ValueError('Full974-record replay required')
    maximum=0.
    for start in range(0,974,4):
        forces,_=arm['forces'](part.positions[start:start+4],species=arm['species'],record_id='adapter-replay')
        if not np.allclose(forces,expected[start:start+4],rtol=1e-10,atol=1e-10):raise ValueError('Batch adapter replay differs')
        maximum=max(maximum,float(np.max(np.abs(forces-expected[start:start+4]))))
    representatives=[next(i for i,rid in enumerate(part.record_ids) if part.memberships[rid]['groups']==[group]) for group in GROUPS]
    for i in representatives:
        forces,_=arm['forces'](part.positions[i],species=arm['species'],record_id=part.record_ids[i])
        if not np.allclose(forces,expected[i],rtol=1e-10,atol=1e-10):raise ValueError('Single-molecule adapter replay differs')
        maximum=max(maximum,float(np.max(np.abs(forces-expected[i]))))
    return {'status':'verified','record_count':974,'single_record_ids':[part.record_ids[i] for i in representatives],
        'maximum_absolute_force_difference':maximum,'rtol':1e-10,'atol':1e-10,'checkpoint_identity':arm['checkpoint_identity']}


def worker(records_file,stage,model_id,output):
    assert_imports();record=read_json(records_file);_verify_records(record)
    starts,refs,policy,binding,manifest,classification,seen=_inputs(record)
    if stage=='control':
        root=fresh_directory(output);control=geometry_controls(starts,refs,classification,policy)
        atomic_json(root/'control.json',control);atomic_json(root/'input-binding.json',binding);freeze_start_set_hash(starts,root)
    else:
        from evaluation.e3x_adapter import e3x_checkpoint_arm
        item=record['models'][model_id]
        arm=e3x_checkpoint_arm(item['run_directory'],item['launch_record'],item['score_directory'],acceptance=item['binding'])
        if stage=='preflight':
            root=fresh_directory(output);result=adapter_replay(arm,item['score_directory'],record['data_directory'],record['evaluation']['split_identity']);atomic_json(root/'replay.json',result)
        else:run_model_pass(starts,arm,record['evaluation']['relaxation'],classification,refs,policy,manifest,seen_minimum=seen,output=output)
    _verify_records(record)


def run_validation(records_file,output,approval_file):
    assert_imports();record=read_json(records_file);approval=read_json(approval_file)
    if approval['status']!='accepted_for_validation_recovery' or digest(Path(records_file).read_bytes())!=approval['records_sha256']:raise ValueError('Reviewed validation records required')
    reviews=approval['review_sha256']
    if len(reviews)!=2 or len({str(Path(p).resolve()) for p in reviews})!=2 or len(set(reviews.values()))!=2:raise ValueError('Two independent record reviews required')
    for path,h in reviews.items():
        if digest(Path(path).read_bytes())!=h:raise ValueError('Review binding mismatch')
    _verify_records(record)
    if str(Path(output).resolve())!=record['output']:raise ValueError('Reviewed output differs')
    margin=record['phase_reconciliation']['margin_seconds']
    if ('start_binding' not in record or isinstance(margin,bool) or not isinstance(margin,(int,float))
            or not math.isfinite(margin) or margin<0):raise ValueError('Frozen starts/resource reconciliation required')
    require_storage(output,record['required_free_storage_bytes']);root=fresh_directory(output)
    deadline=time.perf_counter()+record['global_wall_seconds'];stages=[]
    schedule=[('control',None,record['control_wall_seconds'])]+[('preflight',cid,120.) for cid in MODEL_ORDER]+[('pass',cid,1800.) for cid in MODEL_ORDER]
    for stage,cid,cap in schedule:
        _verify_records(record);name=stage+('-'+cid if cid else '');started=time.perf_counter()
        entry={'stage':stage,'model_id':cid,'status':'running','automatic_retry':False};stages.append(entry);save_json(root/'stages.json',stages)
        try:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',stage,'--records',str(records_file),'--output',str(root/name)]
            if cid:command+=['--model',cid]
            invoke(command,root/(name+'.txt'),min(cap,remaining_seconds(deadline)))
            if stage=='preflight' and read_json(root/name/'replay.json')['status']!='verified':raise ValueError('Adapter replay did not verify')
            if stage=='pass' and read_json(root/name/'result.json')['complete'] is not True:raise ValueError('Incomplete model pass')
            entry.update(status='completed',elapsed_seconds=time.perf_counter()-started);save_json(root/'stages.json',stages)
        except Exception as exc:
            entry.update(status='failed',elapsed_seconds=time.perf_counter()-started,failure_reason=f'{type(exc).__name__}: {exc}')
            save_json(root/'stages.json',stages);save_json(root/'STOPPED.json',{'failed_stage':entry,'no_automatic_retry':True,'raw_completed_branch_rows_preserved':True});raise
    _verify_records(record);control=read_json(root/'control/control.json');reports={cid:read_json(root/('pass-'+cid)/'result.json') for cid in MODEL_ORDER}
    a2=evaluate_a2(reports['energy-0'],control);summary=recovery_summary(reports,control,a2)
    save_json(root/'A2.json',a2);save_json(root/'SUMMARY.json',summary)
    save_json(root/'COMPLETE.json',{'status':'all_validation_passes_completed_result_reviews_pending','records_sha256':approval['records_sha256'],'A2':a2['status'],'main_test_authorized':False});return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['control','preflight','pass']);p.add_argument('--model');p.add_argument('--records',required=True);p.add_argument('--output',required=True);p.add_argument('--approval');args=p.parse_args()
    if args.worker:worker(args.records,args.worker,args.model,args.output)
    else:run_validation(args.records,args.output,args.approval)

if __name__=='__main__':main()
