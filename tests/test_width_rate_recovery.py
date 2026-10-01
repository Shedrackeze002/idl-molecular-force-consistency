"""Metadata trust, honest one-attempt accounting, and one tiny engineering restore."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import hashlib
import msgpack
import pytest


def h(b):return hashlib.sha256(b).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True))
def source(tmp_path,real=False):
    base=tmp_path/'source';base.mkdir();config={'max_updates':0,'seed':0,'microbatch_size':3,'model':{'features':4,'iterations':1,'radial_basis':4,'max_degree':1,'cutoff_A':5.,'energy_readout_init':'zero'}}
    case={'name':'tiny','features':4,'learning_rate':.001,'wall_seconds':120.};statehash={'params':'p','optimizer':'o'}
    prov={'historical':{200:{'m':2},1000:{'m':3}},'normalization':{},'case':case,'config_identity':h(json.dumps(config,sort_keys=True,separators=(',',':')).encode()),'inputs':{r:{'record_ids':list(range(7))} for r in ('fitting','anchor')}}
    for role in ('fitting','anchor'):(base/f'{role}-inputs.npz').write_bytes(('opaque-'+role).encode())
    (base/'reference.npz').write_bytes(b'opaque-reference')
    raw=msgpack.packb({'payload':b'opaque-state','sha256':h(b'opaque-state'),'control_payload':b'\xc0','control_sha256':h(b'\xc0'),'provenance':json.dumps(prov,sort_keys=True,separators=(',',':'))},use_bin_type=True)
    if real:
        from test_coverage import seven_case
        from force_consistency.coverage_runs import _arrays,_partition_predictions
        from force_consistency.width_rate_runs import canonical_state_hashes
        from force_consistency.data.prepare import array_identity
        from force_consistency.runs import save_checkpoint
        from dataclasses import asdict
        import numpy as np
        part,model,batch,norm,state=seven_case();config['model']=asdict(model.config);statehash=canonical_state_hashes(state)
        inputs={r:{'record_ids':list(part.record_ids),'memberships':part.memberships,'arrays':{k:array_identity(v) for k,v in _arrays(part).items()}} for r in ('fitting','anchor')}
        prov.update(normalization=asdict(norm),model=asdict(model.config),inputs=inputs,config_identity=h(json.dumps(config,sort_keys=True,separators=(',',':')).encode()))
        for role in ('fitting','anchor'):np.savez(base/f'{role}-inputs.npz',**_arrays(part))
        e,f=_partition_predictions(state['params'],part,model,3);np.savez(base/'reference.npz',fitting_energy=e,anchor_energy=e,fitting_forces=f,anchor_forces=f)
        temp=base/'original.msgpack';save_checkpoint(temp,state,prov);raw=temp.read_bytes()
    prov['input_files_sha256']={r:h((base/f'{r}-inputs.npz').read_bytes()) for r in ('fitting','anchor')}
    # Re-embed integer keys after adding the file identities.
    env=msgpack.unpackb(raw,raw=False);env['provenance']=json.dumps(prov,sort_keys=True,separators=(',',':'));raw=msgpack.packb(env,use_bin_type=True)
    write(base/'config.json',config);write(base/'case.json',case);write(base/'provenance.json',prov);(base/'snapshots').mkdir();(base/'snapshots/0.msgpack').write_bytes(raw);write(base/'histories/0.json',[{'update':0}])
    receipt={'update':0,'checkpoint':'snapshots/0.msgpack','history':'histories/0.json','checkpoint_sha256':h(raw),'history_sha256':h((base/'histories/0.json').read_bytes()),'state_hashes':statehash};write(base/'commit.json',receipt)
    contract={'case':case,'config_identity':prov['config_identity'],'update':0,'normalization':prov['normalization'],'state_hashes':statehash,'commit':receipt,'inputs':prov['inputs'],'input_files_sha256':prov['input_files_sha256'],'reference_sha256':h((base/'reference.npz').read_bytes())};write(base/'checkpoint-contract.json',contract)
    c={'expected_update':0,'fitting_frame_count':7,'anchor_frame_count':7,'expected_source_payload_sha256':env['sha256'],'expected_source_control_sha256':env['control_sha256'],'total_wall_seconds':120.,'checkpoint_prediction_atol':1e-10,'recovery_role':'copied_checkpoint_verification_only','new_training_updates':0}
    report={'status':'failed','execution_valid':False,'interpretation_valid':False,'full_pilot':'pending','actual_counts':{'work_counts_complete':False,'known_parent_reference_prediction_frames':14,'unknown_probe_cells':['tiny']},'arms':{'tiny':{'status':'failed','fit_gate':{'accepted':False},'anchor_gate':{'accepted':False},'evaluation_frame_count':14,'oracle_metrics_prediction_frames':14,'oracle_verification':{'prediction_reference_frames':14},'derived_probe_seconds':None}},'shared_initializations':{}}
    return {'config':c,'source_root':base,'source_receipt':receipt,'source_contract':contract,'provenance':json.loads((base/'provenance.json').read_text()),'old_report':report}


def module():from force_consistency import width_rate_recovery as m;return m

def passed(root,v):
    receipt=json.loads((root/'verification-copy/commit.json').read_text());cfg=json.loads((v['source_root']/'config.json').read_text());case=json.loads((v['source_root']/'case.json').read_text())
    r={'status':'passed','update':0,'adam_count':0,'normalization':v['provenance']['normalization'],'state_hashes':receipt['state_hashes'],'checkpoint_sha256':receipt['checkpoint_sha256'],'history_sha256':receipt['history_sha256'],'case':case,'config_identity':v['source_contract']['config_identity'],'fitting_record_count':7,'anchor_record_count':7,'maximum_absolute_error':0.,'tolerance':1e-10,'precision':'float64','template_initializations':1,'template_initializer_frames':3,'prediction_frames':14}
    write(root/'verification-copy/checkpoint-verification.json',r);return r


def test_source_provenance_only_transformation_and_old_reader(tmp_path):
    v=source(tmp_path);m=module();root=tmp_path/'copy';before={str(p.relative_to(v['source_root'])):h(p.read_bytes()) for p in v['source_root'].rglob('*') if p.is_file()}
    transformation=m._stage_copy(v,root,lambda:None);m._audit_copy(v,root,transformation)
    original=msgpack.unpackb((v['source_root']/'snapshots/0.msgpack').read_bytes(),raw=False);copied=msgpack.unpackb((root/'verification-copy/snapshots/0.msgpack').read_bytes(),raw=False)
    assert original['payload']==copied['payload'] and original['control_payload']==copied['control_payload']
    assert copied['provenance']==json.dumps(v['provenance'],sort_keys=True,separators=(',',':')) and original['provenance']!=copied['provenance']
    assert before=={str(p.relative_to(v['source_root'])):h(p.read_bytes()) for p in v['source_root'].rglob('*') if p.is_file()}
    from force_consistency.runs import _read_checkpoint
    with pytest.raises(ValueError,match='provenance'):_read_checkpoint(v['source_root']/'snapshots/0.msgpack',v['provenance'])
    _read_checkpoint(root/'verification-copy/snapshots/0.msgpack',v['provenance'])

@pytest.mark.parametrize('field',['payload','control_payload','provenance'])
def test_corrupt_or_semantically_changed_envelope_rejected(tmp_path,field):
    v=source(tmp_path);raw=v['source_root']/'snapshots/0.msgpack';e=msgpack.unpackb(raw.read_bytes(),raw=False);e[field]='{}' if field=='provenance' else b'corrupt';raw.write_bytes(msgpack.packb(e,use_bin_type=True));v['source_receipt']['checkpoint_sha256']=h(raw.read_bytes())
    with pytest.raises(ValueError):module()._stage_copy(v,tmp_path/'copy',lambda:None)

@pytest.mark.parametrize('path',['config.json','provenance.json','fitting-inputs.npz','reference.npz','histories/0.json','snapshots/0.msgpack','commit.json','checkpoint-contract.json'])
def test_coherent_staged_rehash_cannot_bypass_original_audit(tmp_path,path):
    v=source(tmp_path);m=module();root=tmp_path/'copy';t=m._stage_copy(v,root,lambda:None);target=root/'verification-copy'/path;target.write_bytes(target.read_bytes()+b' ')
    t['copied_files_sha256'][path]=h(target.read_bytes());write(root/'copy-transformation.json',t)
    with pytest.raises(ValueError):m._audit_copy(v,root,t)

@pytest.mark.parametrize('boundary',['snapshots','histories','commit.json','copy-transformation.json'])
def test_interrupted_copy_does_not_publish_complete_receipt(tmp_path,monkeypatch,boundary):
    v=source(tmp_path);m=module();real=m.atomic;root=tmp_path/'copy'
    def fail(path,data):
        if boundary in str(path):raise OSError('injected stage interruption')
        return real(path,data)
    monkeypatch.setattr(m,'atomic',fail)
    with pytest.raises(OSError):m._stage_copy(v,root,lambda:None)
    assert not (root/'copy-transformation.json').exists()


def test_existing_copy_and_unsafe_symlink_rejected(tmp_path):
    v=source(tmp_path);m=module();root=tmp_path/'copy';m._stage_copy(v,root,lambda:None)
    with pytest.raises(ValueError):m._stage_copy(v,root,lambda:None)
    (root/'verification-copy/reference.npz').unlink();(root/'verification-copy/reference.npz').symlink_to(v['source_root']/'reference.npz')
    with pytest.raises(ValueError):m._audit_copy(v,root,json.loads((root/'copy-transformation.json').read_text()))


def test_public_contract_and_all_pins_without_numerics():
    from force_consistency.width_rate_recovery_contract import verify_recovery_sources
    c=json.loads(Path('configs/aspirin-width-rate-recovery.json').read_text());v=verify_recovery_sources(c)
    assert v['source_receipt']['update']==6000
    for field in ('total_wall_seconds','expected_update','expected_source_checkpoint_sha256','recovery_role'):
        bad=copy.deepcopy(c);bad[field]=0
        with pytest.raises(ValueError):verify_recovery_sources(bad)


def test_controller_one_child_and_separate_effort(tmp_path):
    v=source(tmp_path);m=module();calls=[];root=tmp_path/'run'
    def probe(root,timeout):calls.append(timeout);return passed(root,v)
    out=m._recover(v['config'],root,lambda:v,probe)
    assert out['verification_valid'] and out['status']=='passed' and len(calls)==1
    assert out['new_work']=={'training_updates':0,'optimizer_proposals':0,'warmup_frames':0,'template_initializations':1,'template_initializer_frames':3,'prediction_frames':14}
    assert not out['original_effort']['actual_counts']['work_counts_complete'] and not out['original_outcome']['execution_valid']
    with pytest.raises(ValueError):m._recover(v['config'],root,lambda:v,probe)

@pytest.mark.parametrize('kind',['failure','timeout','cap','sourcechange','output'])
def test_controller_failure_work_accounting_and_no_retry(tmp_path,monkeypatch,kind):
    v=source(tmp_path);m=module();calls=[];guards=[0];now=[0.];root=tmp_path/'run'
    def guard():
        guards[0]+=1
        if kind=='cap':now[0]+=121.
        if kind=='sourcechange' and guards[0]>2:raise ValueError('source changed')
        return v
    def probe(root,timeout):
        calls.append(timeout);now[0]+=5.
        if kind=='failure':raise subprocess.CalledProcessError(1,'probe',stderr=b'failure')
        if kind=='timeout':raise subprocess.TimeoutExpired('probe',timeout)
        return passed(root,v)
    if kind=='output':
        real=m.atomic
        def fail(path,data):
            if Path(path).name=='summary.json':raise OSError('final write')
            return real(path,data)
        monkeypatch.setattr(m,'atomic',fail)
    out=m._recover(v['config'],root,guard,probe,clock=lambda:now[0])
    assert not out['verification_valid'] and len(calls)<=1
    if kind in ('failure','timeout'):assert out['probe_work_status']=='unknown' and out['probe_seconds']==5. and out['new_work']['prediction_frames'] is None
    if kind in ('sourcechange','output'):assert out['new_work']['prediction_frames']==14


def test_tiny_real_copied_state_in_fresh_process(tmp_path):
    v=source(tmp_path,real=True);m=module();root=tmp_path/'run'
    def probe(root,timeout):
        child=root/'verification-copy';subprocess.run([sys.executable,'-c','from force_consistency.width_rate_runs import _probe_saved; _probe_saved('+repr(str(child))+')'],check=True,capture_output=True,timeout=timeout)
        return json.loads((child/'checkpoint-verification.json').read_text())
    out=m._recover(v['config'],root,lambda:v,probe)
    assert out['verification_valid'],out.get('failure_reason')


def test_corrupted_copy_never_reaches_child(tmp_path,monkeypatch):
    v=source(tmp_path);m=module();real=m._stage_copy;calls=[]
    def stage(*args):
        t=real(*args);p=Path(args[1])/'verification-copy/reference.npz';p.write_bytes(b'coherently changed');t['copied_files_sha256']['reference.npz']=h(p.read_bytes());write(Path(args[1])/'copy-transformation.json',t);return t
    monkeypatch.setattr(m,'_stage_copy',stage)
    out=m._recover(v['config'],tmp_path/'run',lambda:v,lambda *a:calls.append(a))
    assert not out['verification_valid'] and out['probe_attempts']==0 and calls==[]


def test_started_intent_budget_exhaustion_is_known_not_launched(tmp_path,monkeypatch):
    v=source(tmp_path);m=module();real=m.atomic;now=[0.];calls=[]
    def write_and_exhaust(path,data):
        real(path,data)
        if Path(path).name=='probe-work.json':now[0]=121.
    monkeypatch.setattr(m,'atomic',write_and_exhaust)
    out=m._recover(v['config'],tmp_path/'run',lambda:v,lambda *a:calls.append(a),clock=lambda:now[0])
    assert calls==[] and out['probe_attempts']==0 and out['probe_work_status']=='not_started'
    assert out['new_work']['prediction_frames']==0


def test_metadata_guards_and_envelope_inspection_import_no_numerics():
    code='''import importlib.abc,sys,json
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('jax','flax','optax','e3x','numpy'):raise AssertionError('Numerical import: '+name)
sys.meta_path.insert(0,Deny())
from force_consistency.width_rate_recovery import _expected
from force_consistency.width_rate_recovery_contract import verify_recovery_sources
v=verify_recovery_sources(json.load(open('configs/aspirin-width-rate-recovery.json')));_expected(v)
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_source_reference_swap_after_guard_is_rejected(tmp_path):
    v=source(tmp_path);v['source_files_sha256']={str(p.relative_to(v['source_root'])):h(p.read_bytes()) for p in v['source_root'].rglob('*') if p.is_file()}
    (v['source_root']/'reference.npz').write_bytes(b'swapped after verified guard')
    with pytest.raises(ValueError):module()._stage_copy(v,tmp_path/'copy',lambda:None)
