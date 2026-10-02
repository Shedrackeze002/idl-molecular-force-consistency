"""Real file/control boundaries, including returned failures and once-only guards."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

def api():
    name='train_offset_profile'
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts/train_offset_profile.py')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def test_known_nonfinite_return_is_retained_before_validation(tmp_path):
    m=api();owner=m.cc.DirectoryLease(tmp_path);j=m.Journal(owner,100,10)
    try:
        with pytest.raises(FloatingPointError):j.call('energy',3,lambda:np.array([np.nan]),on_returned=lambda a:owner.write('returned.npy',a.tobytes()))
        assert (tmp_path/'returned.npy').exists()
        assert j.work['energy']=={'attempted_calls':1,'returned_calls':1,'valid_calls':0,'returned_frames':3,'valid_frames':0}
        assert j.pending is None
    finally:owner.close()

def test_unknown_call_and_late_return_fail_closed(tmp_path):
    m=api();owner=m.cc.DirectoryLease(tmp_path);clock=[0.];j=m.Journal(owner,20,5,clock=lambda:clock[0])
    try:
        def interrupted():raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):j.call('gradient',7,interrupted)
        assert j.pending['kind']=='gradient' and j.work['gradient']['returned_calls']==0
        j.pending=None
        def late():clock[0]=16.;return np.array([1.])
        with pytest.raises(TimeoutError):j.call('energy',7,late)
        assert j.work['energy']['returned_frames']==7 and j.work['energy']['valid_calls']==1
        assert j.late_call
    finally:owner.close()

def test_gate_drift_compared_to_frozen_object_and_exact_sources(tmp_path):
    m=api();p=tmp_path/'participant';p.write_text('A')
    gate={'status':'cleared','artifacts':{str(p):m.cc.digest(p)}}
    gp=tmp_path/'gate.json';m.cc.atomic(gp,gate)
    assert m.verify_gate({'prelaunch_gate':str(gp)})==gate
    revised={**gate,'unreviewed':True};m.cc.atomic(gp,revised)
    with pytest.raises(ValueError):m.verify_gate({'prelaunch_gate':str(gp)},gate)
    m.cc.atomic(gp,gate);p.write_text('B')
    with pytest.raises(ValueError):m.verify_gate({'prelaunch_gate':str(gp)})

def test_exclusive_root_and_directory_replacement(tmp_path):
    m=api();root=tmp_path/'run';owner=m.reserve_root(root)
    try:
        with pytest.raises(FileExistsError):m.reserve_root(root)
        root.rename(tmp_path/'original');root.mkdir()
        with pytest.raises(ValueError):owner.json('wrong.json',{})
        assert not (root/'wrong.json').exists()
    finally:owner.close()

def test_public_configuration_and_source_guards_are_pure():
    m=api();c=m.cc.read('configs/aspirin-offset-profile.json');facts=m.verify_inputs(c)
    assert len(facts['pins'])==601 and facts['zero']['parameter_count']==13569
    changed={**c,'max_updates':3}
    with pytest.raises(ValueError):m.verify_inputs(changed)

def small_run(tmp_path):
    import time
    from test_coverage import seven_case
    from dataclasses import asdict
    from force_consistency import late_rate as lr
    from force_consistency.runs import save_checkpoint
    m=api();part,model,full,norm,state=seven_case();source=tmp_path/'source';source.mkdir()
    for role in ('fitting','anchor'):np.savez(source/(role+'-inputs.npz'),**lr._arrays(part))
    from force_consistency.coverage_runs import coverage_metrics
    original={'model':asdict(model.config),'normalization':asdict(norm),'inputs':{r:{'record_ids':list(part.record_ids),'memberships':part.memberships,'arrays':{k:lr.array_identity(v) for k,v in lr._arrays(part).items()}} for r in ('fitting','anchor')},
        'input_files_sha256':{r:m.cc.digest(source/(r+'-inputs.npz')) for r in ('fitting','anchor')},
        'original_baselines':coverage_metrics(state['params'],part,part,model,3),'initial_state_hashes':lr.state_hashes(state)}
    (source/'snapshots').mkdir();save_checkpoint(source/'snapshots/0.msgpack',state,original)
    raw=m.cc._envelope(source/'snapshots/0.msgpack',original)
    facts={'root':str(source),'provenance':original,'pins':{},'zero':m.cc._state_metadata(raw,0,original['normalization'])}
    owner=m.reserve_root(tmp_path/'run');c={'total_wall_seconds':120,'finalization_reserve_seconds':10,'atom_count':3,'microbatch_size':3,'max_updates':3,'oracle_updates':[1,3],'score_updates':[0,1,3],'learning_rate':.0003,'loss_weights':{'energy':.2,'force':1.8},'gate_fraction':.1}
    gate={'status':'private_engineering_fixture','artifacts':{}}
    try:result=m.run_owned(c,facts,gate,{},owner,time.perf_counter())
    finally:owner.close()
    return m,c,facts,gate,tmp_path/'run',result

def test_private_real_model_integration_has_linked_states_and_original_baseline(tmp_path):
    m,c,facts,gate,root,result=small_run(tmp_path)
    assert result['execution_valid'],result
    assert result['updates']==3 and result['accepted_calibrations']==4
    assert result['oracles']['3']['incoming_nonzero_moments']
    assert result['original_baselines']==facts['provenance']['original_baselines']
    p=m.cc.read(root/'provenance.json');commit=m.cc.read(root/'commit.json')
    assert commit==m.cc.read(root/'commits/3.json')
    assert [x['update'] for x in m.cc.read(root/commit['history'])]==[0,1,3]
    for u in range(1,4):
        accepted=m.cc.read(root/'accepted'/ (str(u)+'.json'))
        assert accepted['projection_sha256']==m.cc.digest(root/accepted['projection'])
    assert p['source_zero_state_hashes']==facts['provenance']['initial_state_hashes']

def test_probe_token_duplicate_entry_and_file_substitution(tmp_path):
    import time
    m,c,facts,gate,root,result=small_run(tmp_path);assert result['execution_valid'],result
    a=m.reserve_probe(root,time.perf_counter()+30)
    with pytest.raises(FileExistsError):m.reserve_probe(root,time.perf_counter()+30)
    with pytest.raises(ValueError):m.claim_probe_entry(root,'wrong-token')
    assert not (root/'probe-entry.json').exists()
    m.claim_probe_entry(root,a['token'])
    with pytest.raises(FileExistsError):m.claim_probe_entry(root,a['token'])
    (root/'predictions/3.npz').write_bytes(b'substituted')
    with pytest.raises(ValueError):m.claim_probe_entry(root,a['token'])

def test_child_nonzero_return_retains_output_and_cannot_validate_private_root(tmp_path):
    import time
    m,c,facts,gate,root,result=small_run(tmp_path);assert result['execution_valid'],result
    a=m.reserve_probe(root,time.perf_counter()+30)
    with pytest.raises(RuntimeError,match='child failed'):m.launch_probe(root,30,a['token'])
    assert (root/'probe-stdout.txt').exists() and (root/'probe-stderr.txt').exists()
    assert m.cc.read(root/'probe-process.json')['exit_code']!=0
    assert not (root/'probe-completion.json').exists()

def test_zero_exit_without_completion_is_rejected(tmp_path,monkeypatch):
    import subprocess
    m=api()
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,b'returned',b''))
    with pytest.raises(FileNotFoundError):m.launch_probe(tmp_path,1,'token')
    assert (tmp_path/'probe-stdout.txt').read_bytes()==b'returned'
    assert m.cc.read(tmp_path/'probe-process.json')['exit_code']==0

def test_symlink_directory_replacement_rejects_write(tmp_path):
    m=api();root=tmp_path/'run';owner=m.reserve_root(root);root.rename(tmp_path/'saved');root.symlink_to(tmp_path/'saved')
    try:
        with pytest.raises(ValueError):owner.write('output',b'unsafe')
        assert not (tmp_path/'saved/output').exists()
    finally:owner.close()
