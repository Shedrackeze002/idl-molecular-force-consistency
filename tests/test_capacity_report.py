"""Saved-artifact audits: meaningful numerical evidence and adversarial receipts."""
from contextlib import contextmanager
import copy
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pytest


def report_module():return importlib.import_module('force_consistency.capacity_report')


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    from test_capacity_runs import tiny_setup,real_tiny_probe,runner
    from force_consistency import capacity_contract as cc
    from force_consistency.runs import code_provenance
    root=tmp_path_factory.mktemp('capacity-saved-pair');c,case,support=tiny_setup()
    c.update(cases=[case,{**case,'name':'width6','features':6}],total_wall_seconds=120,
             evidence_log=str(root.parent/'evidence.md'),run_directory=str(root))
    Path(c['evidence_log']).write_text('existing evidence\n')
    cc.atomic(root/'config.json',c);cc.atomic(root/'code.json',support['code'])
    results={}
    for case in c['cases']:
        out=runner()._run_case(c,case,support,root/case['name'],time.perf_counter()+120,probe_runner=real_tiny_probe)
        assert out['execution_valid'],out.get('failure_reason');results[case['name']]=out
    cc.atomic(root/'cases.json',results)
    cc.atomic(root/'completion.json',{'execution_valid':True,'elapsed_seconds':1,'failure_reason':None,'case_names':list(results)})
    m=report_module();m.freeze_inventory(root)
    return root,c


@contextmanager
def changed(path,fn):
    original=path.read_bytes()
    try:
        value=json.loads(original);fn(value);path.write_text(json.dumps(value));yield
    finally:path.write_bytes(original)


def test_saved_pair_metrics_gates_work_and_width_specific_initial_states(pair):
    root,c=pair;m=report_module();r=m._build_report(root,production=False,allow_pending_finalization=True)
    assert r['execution_valid'],r['verification_errors']
    assert r['updates']==6 and r['frame_exposures']==42 and r['verified_persisted_updates']==6
    assert not r['any_learning_accepted']
    a,b=r['cases'].values();assert a['initial_state_hashes']!=b['initial_state_hashes']
    assert a['history'][0]['metrics']==b['history'][0]['metrics']
    assert a['child_work']['prediction_frames']==14 and a['distinct_prediction_structures']==7
    assert a['work']['oracle_fullbatch_gradient_frames']==14
    assert a['work']['oracle_accumulated_gradient_frames']==14
    for name,case in r['cases'].items():
        with np.load(root/name/'reference.npz') as pred,np.load(root/name/'fitting-inputs.npz') as inp:
            energy=float(np.mean((pred['fitting_energy']-inp['energy_residual'])**2))
            forces=float(np.mean((pred['fitting_forces']-inp['forces'])**2))
        assert case['endpoint_metrics']['fitting']['energy_mse_eV2']==pytest.approx(energy,abs=1e-12)
        assert case['endpoint_metrics']['fitting']['force_mse_eV2_per_A2']==pytest.approx(forces,abs=1e-12)
    contrast=r['endpoint_contrasts']['fitting']['energy_mse_eV2']
    assert contrast['wider_minus_narrower']==pytest.approx(b['endpoint_metrics']['fitting']['energy_mse_eV2']-a['endpoint_metrics']['fitting']['energy_mse_eV2'])


@pytest.mark.parametrize('file,damage',[
 ('case.json',lambda v:v.update(features=6)),
 ('provenance.json',lambda v:v['model'].update(features=6)),
 ('summary.json',lambda v:v.update(scoring_prediction_frames=0)),
 ('summary.json',lambda v:v.update(child_work=None)),
 ('summary.json',lambda v:v.update(updates=1)),
 ('oracle-3.json',lambda v:v.update(incoming_nonzero_moments=False)),
 ('oracle-3.json',lambda v:v.update(independent_numpy_proposals=0)),
 ('checkpoint-verification.json',lambda v:v.update(maximum_absolute_error=-1)),
 ('probe-entry.json',lambda v:v.update(attempt_sha256='0'*64)),
 ('probe-completion.json',lambda v:v.update(entry_sha256='0'*64)),
 ('histories/3.json',lambda v:v.pop(1))])
def test_case_tampering_removes_complete_contrasts_even_if_inventory_is_rehashed(pair,file,damage):
    root,c=pair;m=report_module();inventory=(root/'raw-artifacts.json').read_bytes()
    try:
        with changed(root/'width4'/file,damage):
            m.freeze_inventory(root)
            r=m._build_report(root,production=False,allow_pending_finalization=True)
            assert not r['execution_valid'] and r['endpoint_contrasts'] is None
            assert r['verification_errors']
    finally:(root/'raw-artifacts.json').write_bytes(inventory)


def test_nonfinite_saved_reference_is_rejected(pair):
    root,c=pair;m=report_module();path=root/'width4/reference.npz';saved=path.read_bytes();inventory=(root/'raw-artifacts.json').read_bytes()
    try:
        with np.load(path) as d:values={k:np.array(v) for k,v in d.items()}
        values['fitting_energy'][0]=np.nan;np.savez(path,**values);m.freeze_inventory(root)
        r=m._build_report(root,production=False,allow_pending_finalization=True)
        assert not r['execution_valid'] and r['endpoint_contrasts'] is None
    finally:path.write_bytes(saved);(root/'raw-artifacts.json').write_bytes(inventory)


def test_zero_denominator_is_named_and_difference_preserved():
    m=report_module();assert m.metric_contrast(0,2)=={'wider_minus_narrower':2,'wider_divided_by_narrower':None,'ratio_unavailable_reason':'narrower denominator is zero'}


def test_late_failure_retains_completed_child_and_receipt_work(pair):
    root,c=pair;m=report_module();inventory=(root/'raw-artifacts.json').read_bytes()
    try:
        with changed(root/'width4/summary.json',lambda v:v.update(execution_valid=False,status='failed',failure_reason='late source failure')):
            m.freeze_inventory(root);r=m._build_report(root,production=False,allow_pending_finalization=True)
            a=r['cases']['width4'];assert not r['execution_valid'] and a['verified_persisted_updates']==3
            assert a['child_work']['prediction_frames']==14 and a['child_work']['template_initializations']==1
    finally:(root/'raw-artifacts.json').write_bytes(inventory)


def test_missing_case_is_unknown_not_zero(pair):
    root,c=pair;m=report_module();p=m._partial_case(root/'absent',c,None)
    assert p['updates'] is None and p['verified_persisted_updates'] is None and p['pending_update_unknown']
    assert p['child_work'] is None


def test_closure_byte_regeneration_and_post_marker_overrun(pair,tmp_path,monkeypatch):
    root,c=pair;m=report_module();cc=m.cc
    monkeypatch.setattr(m,'REPORT_JSON',tmp_path/'report.json');monkeypatch.setattr(m,'REPORT_MD',tmp_path/'report.md')
    r=m._build_report(root,production=False,allow_pending_finalization=True);m.write_report(r)
    monkeypatch.setattr(m.time,'perf_counter',lambda:10.)
    owner=cc.DirectoryLease(root)
    try:m.finalize(c,owner,0,r)
    finally:owner.close()
    rebuilt=m._build_report(root,production=False)
    assert rebuilt==r and m.verify_saved_report(rebuilt)
    # Editing a later evidence suffix remains allowed; the committed prefix is bound.
    with Path(c['evidence_log']).open('a') as f:f.write('later research\n')
    assert m._build_report(root,production=False)['execution_valid']
    # Marker persistence may cross cap even if substantive writes were within it.
    values=iter([10.,10.,121.,121.,121.]);monkeypatch.setattr(m.time,'perf_counter',lambda:next(values,121.))
    owner=cc.DirectoryLease(root)
    try:m.finalize(c,owner,0,r)
    finally:owner.close()
    assert not r['execution_valid'] and r['endpoint_contrasts'] is None
    assert not cc.read(root/'finalization.json')['within_global_cap']
    assert m._build_report(root,production=False)==r
    assert m.verify_saved_report(r)
    (root/'finalization.json').unlink()


def test_controller_failure_blocks_later_case_and_retains_known_first_work(tmp_path,monkeypatch):
    from test_capacity_runs import tiny_setup
    from force_consistency import capacity_runs as runs,capacity_contract as cc
    m=report_module();c,case,support=tiny_setup();root=tmp_path/'once'
    c.update(cases=[case,{**case,'name':'width6','features':6}],run_directory=str(root),total_wall_seconds=120,
             evidence_log=str(tmp_path/'evidence.md'))
    Path(c['evidence_log']).write_text('prior\n');monkeypatch.setattr(cc,'validate_contract',lambda c:None)
    monkeypatch.setattr(cc,'verify_sources',lambda c:{'fixture':'no fullsize source'})
    monkeypatch.setattr(runs,'load_support',lambda c:{**support,'source_facts':{}})
    calls=[]
    def fail(c,case,support,path,deadline):
        calls.append(case['name']);child=support['_root_owner'].child(case['name'],create=True)
        result={'case':case,'execution_valid':False,'status':'failed','failure_reason':'known failure','updates':2,
                'attempted_updates':3,'persisted_updates':1,'frame_exposures':14,'pending_update_unknown':True}
        try:child.json('summary.json',result)
        finally:child.close()
        return result
    monkeypatch.setattr(runs,'_run_case',fail)
    monkeypatch.setattr(m,'REPORT_JSON',tmp_path/'report.json');monkeypatch.setattr(m,'REPORT_MD',tmp_path/'report.md')
    r=m.execute(c)
    assert calls==['width4'] and not r['execution_valid']
    assert r['cases']['width4']['updates']==2
    assert r['cases']['width6']['status']=='blocked' and r['cases']['width6']['verified_persisted_updates']==0
    assert not (root/'width6/probe-attempt.json').exists()
    with pytest.raises(FileExistsError):m.execute(c)


def test_missing_local_summary_retains_controller_returned_computation_and_child_work(pair):
    root,c=pair;m=report_module();path=root/'width4/summary.json';saved=path.read_bytes();inventory=(root/'raw-artifacts.json').read_bytes()
    try:
        path.unlink();m.freeze_inventory(root);r=m._build_report(root,production=False,allow_pending_finalization=True)
        a=r['cases']['width4'];assert not r['execution_valid']
        assert a['updates']==3 and a['attempted_updates']==3 and a['frame_exposures']==21
        assert a['reported_child_work']['prediction_frames']==14
    finally:path.write_bytes(saved);(root/'raw-artifacts.json').write_bytes(inventory)


def test_incomplete_child_phase_preserves_known_template_and_unknown_predictions(tmp_path):
    m=report_module();c={'max_updates':3};path=tmp_path/'child';path.mkdir()
    m.cc.atomic(path/'probe-phase.json',{'phase':'predictions','status':'started','known_template_initializations':1,'known_template_frames':3})
    r=m._partial_case(path,c,{'updates':3,'probe_status':'unknown'})
    assert r['child_phase']['phase']=='predictions'
    assert r['partial_child_work']['reported_template_initializations']==1
    assert r['partial_child_work']['reported_template_initializer_frames']==3
    assert r['partial_child_work']['prediction_frames'] is None
