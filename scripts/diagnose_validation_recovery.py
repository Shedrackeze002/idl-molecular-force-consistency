"""Fixed saved-endpoint geometry and force diagnosis; never retrain or relax."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from e3x_phase_common import ROOT,assert_imports,read_json,save_json,digest,encoded,fresh_directory
sys.path.insert(0,str(ROOT))
from benchmark_e3x_budget import invoke
from launch_e3x_candidates import remaining_seconds,require_storage
from run_validation_recovery import MODEL_ORDER,EXPECTED_KEYS,atomic_json,_verify_records
from force_consistency.data.source import verify_archive
from force_consistency.data.xyz import parse_xyz
from force_consistency.data.geometry import align_records,chemical_graph,descriptor
from force_consistency.data.matching import classify_endpoint
from force_consistency.data.types import PreparationPolicy

TOLERANCE=.01
PRIOR_ACCEPTANCE='b31845316809ce1ce6c5040d99198ad5b967cac22b056ea726d1fd8fa6c027f5'


def force_metrics(positions,forces):
    p=np.asarray(positions,dtype=np.float64);f=np.asarray(forces,dtype=np.float64)
    if p.ndim!=2 or not len(p) or p.shape[-1]!=3 or f.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError('Finite matching nonempty (N,3) arrays required')
    return {'max_abs_component_eV_per_A':float(np.max(np.abs(f))),
        'rms_component_eV_per_A':float(np.sqrt(np.mean(f*f))),
        'net_force_eV_per_A':np.sum(f,axis=0).tolist(),
        'centroid_torque_eV':np.sum(np.cross(p-np.mean(p,axis=0),f),axis=0).tolist()}


def entries_digest(entries):return digest(encoded(entries))


def validate_entries(entries,species):
    if len(entries)!=31:raise ValueError('Exactly31 fixed structures per model required')
    expected=['start']*10+['endpoint']*10+['reference']*11
    if [e['role'] for e in entries]!=expected:raise ValueError('Fixed10/10/11 roles/order required')
    for role in ('start','endpoint'):
        subset=[e for e in entries if e['role']==role]
        if {e['branch_key'] for e in subset}!=EXPECTED_KEYS:raise ValueError('Validation support differs')
    if len({(e['role'],e['record_id']) for e in entries})!=31:raise ValueError('Duplicate diagnostic identity')
    for entry in entries:
        p=np.asarray(entry['positions'],dtype=np.float64)
        if tuple(entry['species'])!=tuple(species) or p.shape!=(len(species),3) or not np.isfinite(p).all():raise ValueError('Diagnostic coordinates/species differ or nonfinite')
        if entry['role']=='reference':force_metrics(p,entry['reference_forces'])
        elif entry['reference_forces'] is not None:raise ValueError('Only exact reference labels may be used as ground truth')
        if entry['role']=='endpoint' and type(entry['recorded_converged']) is not bool:raise ValueError('Recorded convergence must remain boolean')


def allowed_member_names():
    base='Landscape17/aspirin/'
    return {base+'SteepestDescent/'+key.split('/',1)[1].rsplit('/',1)[0]+'/coords'+key.rsplit('/',1)[1]+'_000.xyz' for key in EXPECTED_KEYS}|{base+f'Conformations/coordsmin_{i}.xyz' for i in range(11)}


def read_selected_archive(archive,members):
    if set(members)!=allowed_member_names():raise ValueError('Exact21 selected validation/reference members required')
    verify_archive(Path(archive));records={}
    with zipfile.ZipFile(archive) as z:
        for name,h in sorted(members.items()):
            data=z.read(name)
            if digest(data)!=h:raise ValueError('Bound archive member changed: '+name)
            records[name]=parse_xyz(data.decode('utf-8'),source_path=name,molecule='aspirin')
    return records


def geometry_comparison(structure,own,references,policy):
    own_result=classify_endpoint(structure,{own.record_id:own},policy)
    pool=classify_endpoint(structure,references,policy)
    alignments={}
    for key,ref in sorted(references.items()):
        a=align_records(structure,ref,policy)
        alignments[key]={'search_complete':bool(a.search_complete),'reason':a.reason,
            'rmsd_A':float(a.rmsd) if a.search_complete and np.isfinite(a.rmsd) else None,
            'graph_isomorphic':False if a.reason=='graph_mismatch' else (True if np.isfinite(a.rmsd) else None)}
    n=len(structure.species);indices=np.triu_indices(n,1)
    distances=lambda p:np.linalg.norm(np.asarray(p)[:,None,:]-np.asarray(p)[None,:,:],axis=-1)[indices]
    delta=distances(structure.positions)-distances(own.positions)
    edges=lambda r:{tuple(sorted(e)) for e in chemical_graph(r,policy).edges}
    end_edges=edges(structure);own_edges=edges(own)
    return {'own':own_result,'pool':pool,'own_alignment':alignments[own.record_id],'pool_alignments':alignments,
        'own_descriptor_distance':float(np.linalg.norm(descriptor(structure)-descriptor(own))),
        'own_descriptor_bound':n*policy.endpoint_tolerance+1e-10,
        'fixed_order_pair_distance_max_abs_change_A':float(np.max(np.abs(delta))),
        'fixed_order_pair_distance_mean_abs_change_A':float(np.mean(np.abs(delta))),
        'fixed_order_edges_added':[list(e) for e in sorted(end_edges-own_edges)],
        'fixed_order_edges_removed':[list(e) for e in sorted(own_edges-end_edges)],
        'interpretation':'Fixed stored atom order; not a chemical bond-breaking claim. Nonconverged geometry is diagnostic only; official category unchanged.'}


def history_availability(row):
    audit=row.get('_audit_record',{});missing=[k for k in ('steps','force_history','final_max_force_component_eV_per_A') if k not in audit]
    return {'status':'unavailable' if missing else 'available','missing_fields':missing,
        'trajectory_reconstructed':False,'explanation':'Endpoints/counts cannot reconstruct unretained history'}


def probe_batches(arm,entries,expected_starts,output,*,input_binding=None):
    validate_entries(entries,arm['species'])
    starts=[e for e in entries if e['role']=='start']
    if set(expected_starts)!={e['record_id'] for e in starts}:raise ValueError('Exact ten saved start predictions required')
    for e in starts:force_metrics(e['positions'],expected_starts[e['record_id']])
    root=fresh_directory(output);records=[];calls=attempted=completed=0;started=time.perf_counter()
    if input_binding is not None:atomic_json(root/'input-binding.json',input_binding)
    try:
        for number,offset in enumerate(range(0,31,4)):
            batch=entries[offset:offset+4];p=np.asarray([e['positions'] for e in batch],dtype=np.float64)
            calls+=1;attempted+=len(batch)
            atomic_json(root/f'attempt-{number:03d}.json',{'supplier_calls_attempted':calls,'molecular_evaluations_attempted':attempted,'entry_digests':[entries_digest([e]) for e in batch]})
            forces,_=arm['forces'](p,species=arm['species'],record_id=f'diagnostic-batch-{number}')
            forces=np.asarray(forces,dtype=np.float64)
            if forces.shape!=p.shape or not np.isfinite(forces).all():raise ValueError('Invalid diagnostic force output')
            rows=[];parity=True
            for e,f in zip(batch,forces):
                metrics=force_metrics(e['positions'],f);reference=e['reference_forces']
                row={**e,'predicted_forces_eV_per_A':f.tolist(),'force_metrics':metrics,
                    'checkpoint_identity':arm['checkpoint_identity'],'entry_sha256':entries_digest([e]),
                    'label_force_metrics':force_metrics(e['positions'],reference) if reference is not None else None,
                    'prediction_error_metrics':force_metrics(e['positions'],f-np.asarray(reference)) if reference is not None else None,
                    'start_prediction_parity':None,'start_prediction_max_abs_difference':None,
                    'endpoint_force_rule_matches_recorded_flag':None}
                if e['role']=='start':
                    expected=expected_starts[e['record_id']];ok=bool(np.allclose(f,expected,rtol=1e-10,atol=1e-10));parity&=ok
                    row.update(start_prediction_parity=ok,start_prediction_max_abs_difference=float(np.max(np.abs(f-expected))))
                if e['role']=='endpoint':row['endpoint_force_rule_matches_recorded_flag']=(metrics['max_abs_component_eV_per_A']<=TOLERANCE)==e['recorded_converged']
                rows.append(row)
            atomic_json(root/f'batch-{number:03d}.json',{'records':rows,'supplier_calls_completed':calls,'molecular_evaluations_completed':completed+len(batch)})
            records.extend(rows);completed+=len(batch)
            if not parity:raise ValueError('Saved start force parity differs; raw outputs preserved')
        result={'complete':True,'records':records,'molecular_evaluations':completed,'supplier_calls':calls,
            'endpoint_flag_discrepancy_count':sum(x['endpoint_force_rule_matches_recorded_flag'] is False for x in records),
            'elapsed_seconds':time.perf_counter()-started,'official_outcomes_unchanged':True}
        atomic_json(root/'result.json',result);return result
    except BaseException as exc:
        atomic_json(root/'INCOMPLETE.json',{'supplier_calls_attempted':calls,'molecular_evaluations_attempted':attempted,'molecular_evaluations_completed':completed,
            'failure_reason':f'{type(exc).__name__}: {exc}','no_automatic_retry':True,'complete':False})
        raise


def load_inputs(record):
    prior=read_json(record['prior_records']);archive_records=read_selected_archive(prior['archive'],record['archive_members'])
    root=Path(prior['output']);classification=read_json(prior['classification']);entries={};branches={}
    refs={rid:archive_records[rid] for rid in prior['start_binding']['reference_ids']}
    if set(refs)!={k for k,v in archive_records.items() if v.kind=='minimum'}:raise ValueError('Reference pool differs')
    for cid in MODEL_ORDER:
        rows=read_json(root/('pass-'+cid)/'result.json')['branches']
        if set(rows)!=EXPECTED_KEYS:raise ValueError('Saved validation branch support differs')
        items=[]
        for role in ('start','endpoint'):
            for key,row in sorted(rows.items()):
                own_id=classification[key]['reference_class'];own=refs[own_id];audit=row['_audit_record'];start=archive_records[row['start_record_id']]
                if (audit['start']['record_id']!=start.record_id or start.frame_index!=0 or audit['reference']['record_id']!=own_id
                    or tuple(audit['reference']['species'])!=own.species or not np.array_equal(audit['reference']['positions'],own.positions)):
                    raise ValueError('Saved start/classified-own-reference identity differs')
                p=start.positions if role=='start' else np.asarray(audit['final_positions'])
                items.append({'role':role,'record_id':start.record_id if role=='start' else row['final_record_id'],
                    'branch_key':key,'positions':p.tolist(),'species':list(start.species),'reference_forces':None,
                    'own_reference_id':own_id,'official_category':row['category'] if role=='endpoint' else None,
                    'recorded_converged':row['converged'] if role=='endpoint' else None})
        for rid,ref in sorted(refs.items()):
            items.append({'role':'reference','record_id':rid,'branch_key':None,'positions':ref.positions.tolist(),
                'species':list(ref.species),'reference_forces':ref.forces.tolist(),'own_reference_id':None,'official_category':None,'recorded_converged':None})
        validate_entries(items,next(iter(refs.values())).species)
        if 'entries' in record and items!=record['entries'][cid]:raise ValueError('Frozen diagnostic input entries differ')
        entries[cid]=items;branches[cid]=rows
    return prior,entries,branches,archive_records,refs,PreparationPolicy(**prior['evaluation']['preparation_policy'])


def verify_record(record):
    if (record['order']!=list(MODEL_ORDER) or record['model_wall_seconds']!=120. or record['geometry_wall_seconds']!=300.
            or record['global_wall_seconds']!=900. or record['automatic_retry'] is not False):raise ValueError('Fixed diagnosis order/caps differ')
    if digest(Path(record['prior_acceptance']).read_bytes())!=PRIOR_ACCEPTANCE:raise ValueError('Prior accepted result identity differs')
    for p,h in record['inputs'].items():
        if digest(Path(p).read_bytes())!=h:raise ValueError('Frozen diagnosis input changed: '+p)
    if digest(Path(__file__).read_bytes())!=record['diagnostic_source_sha256']:raise ValueError('Diagnostic source changed')
    prior=read_json(record['prior_records']);_verify_records(prior)
    if record['archive_members'].keys()!=allowed_member_names():raise ValueError('Exact21 archive members required')
    for cid in MODEL_ORDER:
        validate_entries(record['entries'][cid],record['entries'][cid][0]['species'])
        if entries_digest(record['entries'][cid])!=record['entry_sha256'][cid]:raise ValueError('Diagnostic entry identity changed')
    for spec in record['protected_inventories']:
        inventory=read_json(spec['inventory'])
        for rel,value in inventory.items():
            h=value['sha256'] if isinstance(value,dict) else value
            if digest((Path(spec['root'])/rel).read_bytes())!=h:raise ValueError('Protected historical output changed: '+rel)


def geometry_worker(record,output):
    prior,entries,branches,raw,refs,policy=load_inputs(record);root=fresh_directory(output);rows=[]
    atomic_json(root/'input-binding.json',{'archive_members':record['archive_members'],'test_branch_members_read':0})
    for cid in MODEL_ORDER:
        for item in entries[cid]:
            if item['role']!='endpoint':continue
            key=item['branch_key'];own=refs[item['own_reference_id']]
            endpoint=replace(raw[branches[cid][key]['start_record_id']],record_id=item['record_id'],positions=np.asarray(item['positions']))
            row={'model_id':cid,'branch_key':key,'official_category':item['official_category'],'recorded_converged':item['recorded_converged'],
                'force_evaluation_count':branches[cid][key]['force_evaluation_count'],'step_count':branches[cid][key]['step_count'],'stop_reason':branches[cid][key]['stop_reason'],
                'history':history_availability(branches[cid][key]),'geometry':geometry_comparison(endpoint,own,refs,policy)}
            atomic_json(root/f'row-{len(rows):03d}.json',row);rows.append(row)
    atomic_json(root/'geometry.json',{'complete':True,'rows':rows,'model_force_calls':0,
        'static_limits':['Historical force paths were not retained.','Scale-growth cap is not a force-independent displacement cap.'],
        'official_outcomes_unchanged':True})


def model_worker(record,cid,output):
    from evaluation.e3x_adapter import e3x_checkpoint_arm
    prior,entries,_,_,_,_=load_inputs(record);item=prior['models'][cid]
    # Only validation prediction arrays: label arrays never substitute for endpoint truth.
    with np.load(Path(item['score_directory'])/'best-validation.npz',allow_pickle=False) as z:
        ids=z['record_ids'].tolist();expected={e['record_id']:z['forces'][ids.index(e['record_id'])].copy() for e in entries[cid] if e['role']=='start'}
    arm=e3x_checkpoint_arm(item['run_directory'],item['launch_record'],item['score_directory'],acceptance=item['binding'])
    probe_batches(arm,entries[cid],expected,output,input_binding={'archive_members':record['archive_members'],'entry_sha256':record['entry_sha256'][cid],
        'model_id':cid,'checkpoint_binding':item['binding'],'test_branch_members_read':0,'units':prior['evaluation']['units']})


def diagnosis_summary(geometry,models):
    if not geometry['complete'] or len(geometry['rows'])!=50 or set(models)!=set(MODEL_ORDER):raise ValueError('Incomplete diagnosis support')
    summary={}
    for cid in MODEL_ORDER:
        result=models[cid]
        if result['complete'] is not True or result['molecular_evaluations']!=31 or result['supplier_calls']!=8:raise ValueError('Incomplete fixed force probe')
        validate_entries(result['records'],result['records'][0]['species'])
        refs=[x for x in result['records'] if x['role']=='reference'];ends=[x for x in result['records'] if x['role']=='endpoint']
        summary[cid]={'reference_count':11,'reference_prediction_above_force_threshold':sum(x['force_metrics']['max_abs_component_eV_per_A']>TOLERANCE for x in refs),
            'reference_labels_above_force_threshold':sum(x['label_force_metrics']['max_abs_component_eV_per_A']>TOLERANCE for x in refs),
            'endpoint_flag_discrepancy_count':sum(x['endpoint_force_rule_matches_recorded_flag'] is False for x in ends),
            'maximum_start_parity_difference':max(x['start_prediction_max_abs_difference'] for x in result['records'] if x['role']=='start'),
            'maximum_net_force_norm_eV_per_A':max(float(np.linalg.norm(x['force_metrics']['net_force_eV_per_A'])) for x in result['records']),
            'maximum_centroid_torque_norm_eV':max(float(np.linalg.norm(x['force_metrics']['centroid_torque_eV'])) for x in result['records'])}
    return {'status':'complete_diagnosis_result_reviews_pending','models':summary,'molecular_force_evaluations':155,'supplier_calls':40,
        'original_A2':'failed','official_outcomes_unchanged':True,'cause_established':False,
        'interpretation':'Exact-reference stationarity mismatch does not exclude a valid nearby point. Endpoint-only force/graph observations do not establish solver causality, conservativity or chemical bond breaking.'}


def run_diagnosis(records_file,output,approval_file):
    assert_imports();record=read_json(records_file);approval=read_json(approval_file)
    reviews=approval['review_sha256']
    if (approval['status']!='accepted_for_diagnosis' or digest(Path(records_file).read_bytes())!=approval['records_sha256'] or len(reviews)!=2
            or len({str(Path(p).resolve()) for p in reviews})!=2 or len(set(reviews.values()))!=2):raise ValueError('Two exact independent record reviews required')
    for p,h in reviews.items():
        if digest(Path(p).read_bytes())!=h:raise ValueError('Reviewed record binding changed')
    verify_record(record)
    margin=record['margin_seconds']
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin) or margin<0 or str(Path(output).resolve())!=record['output']:raise ValueError('Reviewed output/resource reconciliation required')
    require_storage(output,1024**3);root=fresh_directory(output);deadline=time.perf_counter()+900.;stages=[]
    for cid,cap in [(None,300.)]+[(cid,120.) for cid in MODEL_ORDER]:
        verify_record(record);name='geometry' if cid is None else cid;entry={'stage':name,'status':'running','automatic_retry':False};stages.append(entry);save_json(root/'stages.json',stages);started=time.perf_counter()
        try:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','geometry' if cid is None else 'model','--records',str(records_file),'--output',str(root/name)]
            if cid is not None:command+=['--model',cid]
            invoke(command,root/(name+'.txt'),min(cap,remaining_seconds(deadline)))
            name_result='geometry.json' if cid is None else 'result.json'
            if read_json(root/name/name_result)['complete'] is not True:raise ValueError('Incomplete worker result')
            entry.update(status='completed',elapsed_seconds=time.perf_counter()-started);save_json(root/'stages.json',stages)
        except BaseException as exc:
            entry.update(status='failed',elapsed_seconds=time.perf_counter()-started,failure_reason=f'{type(exc).__name__}: {exc}')
            save_json(root/'stages.json',stages);atomic_json(root/'STOPPED.json',{'failed_stage':entry,'no_automatic_retry':True,'completed_raw_batches_preserved':True});raise
    verify_record(record);summary=diagnosis_summary(read_json(root/'geometry/geometry.json'),{cid:read_json(root/cid/'result.json') for cid in MODEL_ORDER})
    atomic_json(root/'SUMMARY.json',summary);atomic_json(root/'COMPLETE.json',{'status':'all_diagnostics_completed_result_reviews_pending','records_sha256':approval['records_sha256'],'main_test_access':False});return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['geometry','model']);p.add_argument('--model',choices=MODEL_ORDER);p.add_argument('--records',required=True);p.add_argument('--output',required=True);p.add_argument('--approval');a=p.parse_args()
    if a.worker:
        assert_imports();r=read_json(a.records);verify_record(r)
        if a.worker=='geometry':geometry_worker(r,a.output)
        else:model_worker(r,a.model,a.output)
        verify_record(r)
    else:run_diagnosis(a.records,a.output,a.approval)

if __name__=='__main__':main()
