"""Fixed validation replay and reference-start controls, with no solver edits."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from e3x_phase_common import ROOT, digest, encoded, read_json, assert_imports
sys.path.insert(0, str(ROOT))
from recovery_trace import TraceBudget, TraceSupplier, reconstruct_trace, summarize_trace
from run_validation_recovery import MODEL_ORDER, safe_json, atomic_json
from diagnose_validation_recovery import verify_record, load_inputs
from evaluation.relax import relax
from evaluation.runner import _arm_force_supplier
from evaluation.outcomes import classify_outcome
from force_consistency.data.matching import classify_endpoint
from force_consistency.data.geometry import align_records

OWN_IDS = [f'Landscape17/aspirin/Conformations/coordsmin_{i}.xyz' for i in (1,3,6,7,8,9)]
REPLAYS = [('direct-0','aspirin/18/plus'),('consistency-0.1-0','aspirin/18/plus'),
           ('energy-0','aspirin/18/plus'),('energy-0','aspirin/13/plus')]
NEW_SOURCES = ('recovery_trace.py','trace_validation_recovery.py')


def fixed_schedule(own_ids):
    if sorted(own_ids) != OWN_IDS:
        raise ValueError('Exactly six frozen unique own references required')
    tasks=[{'kind':'replay','model_id':m,'branch_key':b} for m,b in REPLAYS]
    return tasks+[{'kind':'reference_control','model_id':m,'reference_id':r}
                  for m in MODEL_ORDER for r in OWN_IDS]


def reproduction_gate(result, category, original):
    for key in ('force_evaluation_count','step_count','converged','budget_exhausted','forces_finite'):
        if result[key] != original[key]:
            raise ValueError('Replay differs on '+key)
    if result['failure_reason'] != original['_audit_record']['failure']['failure_reason']:
        raise ValueError('Replay differs on failure_reason')
    if category != original['category']:
        raise ValueError('Replay differs on official category')
    a=np.asarray(result['final_positions']);b=np.asarray(original['_audit_record']['final_positions'])
    if a.shape != b.shape or not np.isfinite(a).all() or float(np.max(np.abs(a-b))) > 1e-8:
        raise ValueError('Replay endpoint differs by more than 1e-8 angstrom')
    return {'status':'reproduced_recorded_outcome','maximum_endpoint_difference_A':float(np.max(np.abs(a-b))),
            'historical_path_bit_identity_established':False}


def score_reference_control(result, own_reference, preparation_policy):
    final=replace(own_reference, record_id=own_reference.record_id+'/control-endpoint',
                  positions=np.asarray(result['final_positions']),forces=np.zeros_like(own_reference.forces))
    geometry=classify_endpoint(final,{own_reference.record_id:own_reference},preparation_policy)
    alignment=align_records(final,own_reference,preparation_policy)
    return {'operational_force_converged_own_match':bool(result['converged'] and result['forces_finite'] and geometry['status']=='matched'),
            'geometry':geometry,'own_alignment':{'search_complete':bool(alignment.search_complete),
                'reason':alignment.reason,'rmsd_A':float(alignment.rmsd) if alignment.search_complete and np.isfinite(alignment.rmsd) else None},
            'stop_reason':result['failure_reason'],
            'final_max_force_component_eV_per_A':result['final_max_force_component_eV_per_A'],
            'nearby_stationary_point_absence_established':False,
            'exact_stationary_point_or_stable_minimum_established':False}


def run_schedule(tasks, execute):
    results=[]
    for task in tasks:
        results.append(execute(task))
    return results


def prepare_package(diagnosis_file, plan_file, plan_acceptance_file, output_root):
    """Read and bind permitted saved inputs; no trained model is queried."""
    diagnostic=read_json(diagnosis_file);verify_record(diagnostic)
    prior,entries,branches,raw,refs,policy=load_inputs(diagnostic)
    own_ids=sorted({e['own_reference_id'] for e in entries['energy-0'] if e['role']=='start'})
    tasks=fixed_schedule(own_ids)
    plan_acceptance=read_json(plan_acceptance_file)
    if plan_acceptance['status'] != 'AMENDED_PLAN_ACCEPTED_TWO_INDEPENDENT_REVIEWS' or digest(Path(plan_file).read_bytes()) != plan_acceptance['plan_sha256']:
        raise ValueError('Reviewed amended plan required')
    inputs={str(Path(x).resolve()):digest(Path(x).read_bytes()) for x in (diagnosis_file,plan_file,plan_acceptance_file)}
    for name,h in plan_acceptance['review_sha256'].items():
        file=Path(plan_acceptance_file).parent/name
        if digest(file.read_bytes()) != h:raise ValueError('Plan review binding differs')
        inputs[str(file.resolve())]=h
    coordinate_bindings={}
    for cid in MODEL_ORDER:
        for e in entries[cid]:
            if e['role']=='start' or e['record_id'] in OWN_IDS:
                coordinate_bindings[cid+'/'+e['record_id']]=digest(encoded({'positions':e['positions'],'species':e['species']}))
    return {'schema':1,'diagnosis_records':str(Path(diagnosis_file).resolve()),'inputs':inputs,
            'new_sources':{str((ROOT/'scripts'/n).resolve()):digest((ROOT/'scripts'/n).read_bytes()) for n in NEW_SOURCES},
            'tasks':tasks,'coordinate_bindings':coordinate_bindings,'own_ids':own_ids,
            'model_bindings':{m:prior['models'][m]['binding'] for m in MODEL_ORDER},
            'procedure':prior['evaluation']['relaxation'],'output':str(Path(output_root).resolve()),
            'max_force_attempts':51000,'numerical_wall_seconds':600.,'finalization_wall_seconds':60.,
            'max_output_bytes':2**30,'automatic_retry':False,'extra_energy_queries':0,'main_test_access':False}


def verify_package(records):
    if (records['tasks'] != fixed_schedule(records['own_ids']) or records['max_force_attempts']!=51000
            or records['numerical_wall_seconds']!=600. or records['finalization_wall_seconds']!=60.
            or records['max_output_bytes']!=2**30 or records['automatic_retry'] is not False
            or records['extra_energy_queries']!=0 or records['main_test_access'] is not False):
        raise ValueError('Fixed diagnostic schedule/caps differ')
    expected={str((ROOT/'scripts'/n).resolve()) for n in NEW_SOURCES}
    if set(records['new_sources']) != expected:raise ValueError('Exact new source inventory required')
    for path,h in {**records['inputs'],**records['new_sources']}.items():
        if digest(Path(path).read_bytes()) != h:raise ValueError('Bound input/source changed: '+path)
    diagnostic=read_json(records['diagnosis_records']);verify_record(diagnostic)
    prior,entries,branches,raw,refs,policy=load_inputs(diagnostic)
    rebuilt={}
    for cid in MODEL_ORDER:
        for e in entries[cid]:
            if e['role']=='start' or e['record_id'] in OWN_IDS:
                rebuilt[cid+'/'+e['record_id']]=digest(encoded({'positions':e['positions'],'species':e['species']}))
    if rebuilt != records['coordinate_bindings'] or {m:prior['models'][m]['binding'] for m in MODEL_ORDER} != records['model_bindings']:
        raise ValueError('Frozen coordinate/checkpoint bindings differ')
    if prior['evaluation']['relaxation']!=records['procedure']:raise ValueError('Frozen procedure differs')
    return prior,entries,branches,raw,refs,policy


def _persist(path, payload, budget):
    if Path(path).exists():raise FileExistsError('Immutable diagnostic output exists')
    data=encoded(safe_json(payload))
    with Path(path).open('xb') as stream:budget.write(stream,data)


def run_package(records, output_root):
    assert_imports()
    started=time.perf_counter()
    prior,entries,branches,raw,refs,policy=verify_package(records)
    root=Path(output_root)
    if str(root.resolve()) != records['output']:raise ValueError('Output identity differs')
    root.mkdir(parents=True,exist_ok=False)
    budget=TraceBudget(51000,started+600.,2**30)
    cache={};completed=[]
    _persist(root/'STARTED.json',{'complete':False,'records_sha256':digest(encoded(records))},budget)
    def execute(task):
        from evaluation.e3x_adapter import e3x_checkpoint_arm
        budget.admit()
        cid=task['model_id'];item=prior['models'][cid]
        if cid not in cache:
            cache[cid]=e3x_checkpoint_arm(item['run_directory'],item['launch_record'],item['score_directory'],acceptance=item['binding'])
        arm=cache[cid]
        if task['kind']=='replay':
            original=branches[cid][task['branch_key']]
            start=raw[original['start_record_id']];own=refs[original['_audit_record']['reference']['record_id']]
            supplier_id=start.record_id+'/relax'
        else:
            own=refs[task['reference_id']];start=own;supplier_id=own.record_id+'/reference-control'
        directory=root/f'{len(completed):02d}-{cid}'
        supplier=_arm_force_supplier(arm,start.species,supplier_id)
        wrapped=TraceSupplier(supplier,directory,1500,budget.deadline,budget.max_bytes,budget=budget)
        result=relax(start.positions,wrapped,records['procedure'],arm=arm)
        # Persist the return before any diagnostic reconstruction can fail.
        _persist(directory/'relaxation.json',result,budget)
        trace=reconstruct_trace(wrapped.queries,result,records['procedure'])
        summary=summarize_trace(trace,5.)
        final=replace(start,record_id=start.record_id+'/trace-final',positions=result['final_positions'],forces=np.zeros_like(start.forces))
        outcome=classify_outcome(final,own,refs,policy,budget_exhausted=result['budget_exhausted'],
                                 converged=result['converged'],forces_finite=result['forces_finite'],
                                 stop_reason=result['failure_reason'],seen_minimum=True)
        reproduction=reproduction_gate(result,outcome['category'],original) if task['kind']=='replay' else None
        control=score_reference_control(result,own,policy) if task['kind']=='reference_control' else None
        _persist(directory/'trace-summary.json',summary,budget)
        payload={'task':task,'category':outcome['category'],'reproduction':reproduction,'reference_control':control,
                 'force_evaluation_count':result['force_evaluation_count'],'step_count':result['step_count'],
                 'stop_reason':result['failure_reason'],'trace_directory':directory.name,
                 'checkpoint_binding':item['binding'],'first_disconnected_trial':summary['first_disconnected_trial'],
                 'first_accepted_disconnected_state':summary['first_accepted_disconnected_state'],
                 'maximum_accepted_path_atom_displacement_from_start_A':summary['maximum_accepted_path_atom_displacement_from_start_A']}
        _persist(directory/'result.json',payload,budget);completed.append(payload)
        print(f"Completed {len(completed)}/34: {cid} {task.get('branch_key',task.get('reference_id'))}",flush=True)
        return payload
    try:
        run_schedule(records['tasks'],execute)
        if time.perf_counter()>budget.deadline:raise RuntimeError('Numerical/reduction deadline reached')
        payload={'complete':True,'results':completed,'supplier_invocations':budget.invocations,
                 'relax_attempts':budget.relax_attempts,
                 'force_attempts':sum(x['force_evaluation_count'] for x in completed),
                 'elapsed_seconds':time.perf_counter()-started,'output_bytes_before_final_receipt':budget.bytes_written,
                 'original_A2':'failed','main_test_access':False,'cause_established':False,'status':'complete_result_review_pending'}
        _persist(root/'RESULT.json',payload,budget)
        return payload
    except BaseException as exc:
        _persist(root/'INCOMPLETE.json',{'complete':False,'completed_calls':len(completed),
                 'supplier_invocations':budget.invocations,'reason':type(exc).__name__+': '+str(exc),
                 'relax_attempts':budget.relax_attempts,
                 'invocation_count_is_in_memory_attempted_not_durable_completion_count':True,
                 'elapsed_seconds':time.perf_counter()-started,'automatic_retry':False},budget)
        raise


def verify_launch_approval(records_file, approval):
    reviews=approval.get('review_sha256',{})
    if (approval.get('status')!='accepted_for_paired_trace'
            or approval.get('records_sha256')!=digest(Path(records_file).read_bytes())
            or len(reviews)!=2 or len(set(reviews.values()))!=2
            or len({str(Path(p).resolve()) for p in reviews})!=2):
        raise ValueError('Two exact independent launch-record reviews required')
    for path,h in reviews.items():
        if digest(Path(path).read_bytes())!=h:raise ValueError('Launch review changed')


def bounded_process(command, logfile, numerical_seconds, finalization_seconds):
    """Enforce the numerical ceiling even if a supplier never returns."""
    started=time.perf_counter();timed_out=False;hard_kill=False
    with Path(logfile).open('xb') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            proc.wait(timeout=numerical_seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=finalization_seconds)
            except subprocess.TimeoutExpired:
                hard_kill=True;os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=finalization_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
    return {'returncode':proc.returncode,'timed_out':timed_out,'hard_kill':hard_kill,
            'elapsed_seconds':time.perf_counter()-started}


def supervise(records_file,approval_file):
    records_file=Path(records_file).resolve();approval_file=Path(approval_file).resolve()
    verify_launch_approval(records_file,read_json(approval_file))
    records=read_json(records_file)
    if records['numerical_wall_seconds']!=600. or records['finalization_wall_seconds']!=60.:
        raise ValueError('Fixed hard watchdog limits required')
    evidence=records_file.parent
    # An interrupted or failed dispatch is still spent; never overwrite it.
    atomic_json(evidence/'DISPATCH_STARTED.json',{'records_sha256':digest(records_file.read_bytes()),
                'automatic_retry':False,'numerical_seconds':600.,'finalization_seconds':60.})
    result=bounded_process([sys.executable,str(Path(__file__).resolve()),str(records_file),
                           '--approval',str(approval_file),'--worker'],evidence/'worker.log',600.,60.)
    atomic_json(evidence/'DISPATCH_ENDED.json',result)
    if result['returncode']!=0 or result['timed_out']:raise RuntimeError('Diagnostic dispatch failed; no retry')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('records')
    parser.add_argument('--approval',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    verify_launch_approval(args.records,read_json(args.approval))
    if args.worker:
        def deadline_signal(signum,frame):raise TimeoutError('Parent numerical deadline; artifact finalization only')
        signal.signal(signal.SIGTERM,deadline_signal)
        records=read_json(args.records);run_package(records,records['output'])
    else:supervise(args.records,args.approval)


if __name__=='__main__':main()
