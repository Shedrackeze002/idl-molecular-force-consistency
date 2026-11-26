"""Real file/control boundaries, including returned failures and once-only guards."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

def api():
    name='train_offset_profile'
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location(name,scripts/'train_offset_profile.py')
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

def small_run(tmp_path,controller=None):
    import time
    from test_coverage import seven_case
    from dataclasses import asdict
    from force_consistency import late_rate as lr
    from force_consistency.runs import save_checkpoint
    m=controller or api();part,model,full,norm,state=seven_case();source=tmp_path/'source';source.mkdir()
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

@pytest.mark.parametrize('late_write',['work.json','summary.json'])
def test_final_durable_writes_are_charged_to_cap(tmp_path,late_write,monkeypatch):
    m=api();clock=[0.];owner=m.cc.DirectoryLease(tmp_path);j=m.Journal(owner,10,2,clock=lambda:clock[0]);real=owner.json
    def delayed(name,value,**kw):
        real(name,value,**kw)
        if name==late_write:clock[0]=11.
    monkeypatch.setattr(owner,'json',delayed)
    try:
        result=m.finalize_summary(j,{'status':'completed','execution_valid':True})
        saved=m.cc.read(tmp_path/'summary.json')
        assert not result['execution_valid'] and not saved['execution_valid']
        assert saved['elapsed_seconds']>=10 and saved['stop_reason']=='wall_budget'
    finally:owner.close()

def test_worker_binds_supervisor_original_gate(tmp_path):
    m=api();c={'prelaunch_gate':str(tmp_path/'gate.json'),'run_directory':'run'};gate={'status':'cleared','artifacts':{}}
    m.cc.atomic(tmp_path/'production-invocation.json',{'ordinal':1,'gate':gate,'config_identity':m.cc.canonical_hash(c),'run_directory':'run'})
    m.verify_invocation(c,gate)
    with pytest.raises(ValueError):m.verify_invocation(c,{**gate,'unreviewed':True})

def test_private_fresh_process_reproduces_real_saved_endpoint(tmp_path):
    import subprocess,time
    m,c,facts,gate,root,result=small_run(tmp_path);assert result['execution_valid'],result
    a=m.reserve_probe(root,time.perf_counter()+60);bundle=tmp_path/'facts.json'
    m.cc.atomic(bundle,{'source':facts,'gate':gate,'root':str(root),'token':a['token']})
    script=tmp_path/'engineering_child.py'
    script.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(Path(__file__).resolve().parents[1]/'scripts'))+')\nfrom train_offset_profile import cc,claim_probe_entry,probe_owned\nb=cc.read(sys.argv[1]);root=Path(b["root"]);entry=claim_probe_entry(root,b["token"]);probe_owned(root,b["source"],b["gate"],cc.read(root/"probe-attempt.json"),entry)\n')
    child=subprocess.run([sys.executable,str(script),str(bundle)],capture_output=True,text=True,timeout=60)
    (tmp_path/'child-stdout.txt').write_text(child.stdout);(tmp_path/'child-stderr.txt').write_text(child.stderr)
    assert child.returncode==0,child.stderr
    r=m.cc.read(root/'checkpoint-verification.json');completion=m.cc.read(root/'probe-completion.json')
    assert r['status']=='passed' and r['prediction_frames']==14
    assert r['maximum_absolute_error']<=1e-10 and r['training_updates']==r['random_initializations']==0
    assert completion['verification_sha256']==m.cc.digest(root/'checkpoint-verification.json')

def test_actual_engine_late_summary_write_invalidates_completion(tmp_path,monkeypatch):
    m=api();clock=[0.];real_journal=m.Journal;real_reserve=m.reserve_root
    def journal(owner,wall,reserve,**kw):
        clock[0]=kw['start'];return real_journal(owner,wall,reserve,clock=lambda:clock[0],**kw)
    def reserve(root):
        owner=real_reserve(root);real_json=owner.json
        def write(name,value,**kw):
            real_json(name,value,**kw)
            if name=='summary.json' and value['execution_valid']:clock[0]+=121
        owner.json=write;return owner
    monkeypatch.setattr(m,'Journal',journal);monkeypatch.setattr(m,'reserve_root',reserve)
    m,c,facts,gate,root,result=small_run(tmp_path,controller=m)
    assert result['updates']==3
    assert not result['execution_valid'] and not m.cc.read(root/'summary.json')['execution_valid']
    assert result['stop_reason']=='wall_budget' and result['elapsed_seconds']>=120
