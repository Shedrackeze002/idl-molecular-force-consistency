"""Replacement boundaries: mutations must fail before numerical work."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

CONFIG = Path('configs/aspirin-width-rate-replacement.json')

def module(name='width_rate_replacement_contract'):
    spec=importlib.util.find_spec('force_consistency.'+name)
    assert spec is not None, 'Replacement boundary not implemented'
    return importlib.import_module('force_consistency.'+name)

@pytest.mark.parametrize('key,value',[
    ('max_updates',6001),('backend','gpu'),('run_directory','runs/aspirin-width-rate-2026-09-30'),
    ('total_wall_seconds',11401),('loss_weights',{'energy':1.,'force':1.}),('extra',True),
])
def test_unreviewed_controls_rejected(key,value):
    config=json.loads(CONFIG.read_text());config[key]=value
    with pytest.raises(ValueError):module().validate_replacement(config)

def test_all_pins_and_ancestry_bind_original_facts():
    out=module().verify_replacement_sources(json.loads(CONFIG.read_text()))
    assert out['config_identity']=='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36'
    assert out['pin_count']==518
    assert len(out['historical'])==63
    assert out['ancestry']['original_status']=='failed'
    assert out['ancestry']['copy_status']=='passed'
    assert out['original_baselines']==json.loads(Path('runs/aspirin-loss-balance-2026-09-30/balanced/history.json').read_text())[0]['metrics']

def test_changed_pinned_file_is_rejected_without_array_import(monkeypatch):
    contract=module();actual=contract.digest
    monkeypatch.setattr(contract,'digest',lambda path:'0'*64 if str(path).endswith('uv.lock') else actual(path))
    with pytest.raises(ValueError,match='source'):contract.verify_replacement_sources(json.loads(CONFIG.read_text()))

@pytest.mark.parametrize('kind',['existing','symlink','race'])
def test_root_creation_never_writes_unowned_root(tmp_path,kind):
    boundary=module();root=tmp_path/'run';other=tmp_path/'other';other.mkdir()
    if kind=='existing':root.mkdir();(root/'keep').write_text('untouched')
    elif kind=='symlink':root.symlink_to(other,target_is_directory=True)
    else:root.mkdir()
    with pytest.raises((ValueError,FileExistsError)):boundary.claim_root(root)
    assert not (root/'invocation.json').exists()
    assert not (other/'invocation.json').exists()
    if kind=='existing':assert (root/'keep').read_text()=='untouched'

def test_original_preflight_receives_only_original_config(monkeypatch):
    replacement=module('width_rate_replacement');old=json.loads(Path('configs/aspirin-width-rate-diagnostic.json').read_text())
    class Engine:
        @staticmethod
        def width_rate_preflight(config,context,save_partial):
            assert config==old
            return {'sources':module().verify_replacement_sources(json.loads(CONFIG.read_text()))['original_sources']}
    monkeypatch.setattr(replacement,'_engine',lambda:Engine)
    out=replacement.replacement_preflight(json.loads(CONFIG.read_text()),None,lambda r:None)
    assert out['sources']['pin_count']==518

def test_bad_public_config_rejected_before_numerical_import(tmp_path):
    bad=json.loads(CONFIG.read_text());bad['backend']='gpu';path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    code='''
import sys, importlib.abc, runpy
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('jax','numpy','flax','optax','e3x'):raise AssertionError('NUMERICAL_IMPORT')
sys.meta_path.insert(0,Deny())
sys.argv=['replacement','--config',sys.argv[1]]
runpy.run_module('force_consistency.width_rate_replacement',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Unreviewed replacement configuration' in result.stderr
    assert 'NUMERICAL_IMPORT' not in result.stderr

@pytest.mark.parametrize('allowance',[0,-1,float('nan'),float('inf')])
def test_no_probe_reserved_without_positive_allowance(tmp_path,allowance,monkeypatch):
    replacement=module('width_rate_replacement')
    class Context:
        def remaining_seconds(self):return allowance
        def check_budget(self):pass
    monkeypatch.setattr(replacement,'audit_replacement_cell',lambda c,r:{'commit':{},'case':{},'contract':{}} ,raising=False)
    with pytest.raises((TimeoutError,ValueError)):replacement.launch_probe(tmp_path,Context())
    assert not (tmp_path/'probe-attempt.json').exists()

@pytest.mark.parametrize('kind',['failed','timeout'])
def test_launcher_preserves_complete_available_logs(tmp_path,kind,monkeypatch):
    replacement=module('width_rate_replacement');(tmp_path/'probe-attempt.json').write_text('{}')
    def fail(*a,**kw):
        if kind=='failed':raise subprocess.CalledProcessError(7,a[0],output=b'A'*10001,stderr=b'B'*10002)
        raise subprocess.TimeoutExpired(a[0],1,output=b'A'*10001,stderr=b'B'*10002)
    monkeypatch.setattr(replacement.subprocess,'run',fail)
    with pytest.raises((subprocess.CalledProcessError,subprocess.TimeoutExpired)):replacement._launch(tmp_path,1,'a'*64)
    assert (tmp_path/'probe-stdout.txt').read_bytes()==b'A'*10001
    assert (tmp_path/'probe-stderr.txt').read_bytes()==b'B'*10002

def test_attempt_claim_is_exclusive_and_bound(tmp_path):
    replacement=module('width_rate_replacement');binding={'ordinal':1,'config_identity':'test','commit':{'update':6000}}
    path=tmp_path/'probe-attempt.json';path.write_text(json.dumps(binding));import hashlib
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):replacement._claim_attempt(tmp_path,'0'*64,binding)
    assert not (tmp_path/'probe-entry.json').exists()
    replacement._claim_attempt(tmp_path,sha,binding)
    with pytest.raises((ValueError,FileExistsError)):replacement._claim_attempt(tmp_path,sha,binding)
    assert json.loads((tmp_path/'probe-entry.json').read_text())['attempt_sha256']==sha

def test_journal_retains_failed_and_blocked_transitions(tmp_path):
    replacement=module('width_rate_replacement')
    replacement.journal(tmp_path,'cell','failed',{'attempted_updates':None})
    replacement.journal(tmp_path,'cell','blocked',{'stop_reason':'previous_failure'})
    rows=[json.loads(x) for x in (tmp_path/'journal.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows]==['failed','blocked']
    assert rows[0]['details']['attempted_updates'] is None
    assert rows[1]['details']['stop_reason']=='previous_failure'

@pytest.mark.parametrize('damage',['config','case','normalization','inputs','reference','history','envelope'])
def test_pure_cell_audit_rejects_coherent_changes(tmp_path,damage):
    replacement=module('width_rate_replacement');root,sources=_staged_endpoint(tmp_path,replacement)
    assert replacement._audit_cell(json.loads(CONFIG.read_text()),root,sources)['commit']['update']==6000
    if damage=='config':
        value=json.loads((root/'config.json').read_text());value['max_updates']=6001;(root/'config.json').write_text(json.dumps(value))
    elif damage=='case':
        for name in ('case.json','provenance.json','checkpoint-contract.json'):
            value=json.loads((root/name).read_text())
            if name=='case.json':value['features']=32
            else:value['case']['features']=32
            (root/name).write_text(json.dumps(value))
    elif damage=='normalization':
        for name in ('provenance.json','checkpoint-contract.json'):
            value=json.loads((root/name).read_text());value['normalization']['energy_offset_eV']+=1;(root/name).write_text(json.dumps(value))
    elif damage=='inputs':
        (root/'fitting-inputs.npz').write_bytes(b'substitute')
        from force_consistency.width_rate_contract import digest
        for name in ('provenance.json','checkpoint-contract.json'):
            value=json.loads((root/name).read_text());value['input_files_sha256']['fitting']=digest(root/'fitting-inputs.npz');(root/name).write_text(json.dumps(value))
    elif damage=='reference':(root/'reference.npz').write_bytes(b'wrong')
    elif damage=='history':
        value=json.loads((root/'histories/200.json').read_text());value[0]['metrics']['fitting']['energy_mse_eV2']+=1;(root/'histories/200.json').write_text(json.dumps(value))
    else:
        import msgpack
        path=root/'snapshots/6000.msgpack';value=msgpack.unpackb(path.read_bytes(),raw=False);value['sha256']='0'*64;path.write_bytes(msgpack.packb(value,use_bin_type=True))
        from force_consistency.width_rate_contract import digest
        receipt=json.loads((root/'commit.json').read_text());receipt['checkpoint_sha256']=digest(path);(root/'commit.json').write_text(json.dumps(receipt))
        contract=json.loads((root/'checkpoint-contract.json').read_text());contract['commit']=receipt;(root/'checkpoint-contract.json').write_text(json.dumps(contract))
    with pytest.raises((ValueError,KeyError)):replacement._audit_cell(json.loads(CONFIG.read_text()),root,sources)

def _staged_endpoint(tmp_path,replacement):
    """Metadata-only fixture, opaque historical payloads; never a numerical result."""
    import hashlib, msgpack
    from force_consistency.width_rate_contract import digest
    c=json.loads(CONFIG.read_text());s=module().verify_replacement_sources(c);root=tmp_path/'endpoint';root.mkdir()
    old=Path('runs/aspirin-width-rate-2026-09-30/width16-lr1e-3');prov=json.loads((old/'provenance.json').read_text())
    prov.update(config_identity='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36',sources=json.loads(json.dumps(s)),source_pin_manifest_identity=c['source_pin_manifest_sha256'])
    for name in ('fitting-inputs.npz','anchor-inputs.npz','reference.npz'):(root/name).write_bytes((old/name).read_bytes())
    for folder in ('snapshots','histories'):(root/folder).mkdir()
    h=json.loads((old/'histories/6000.json').read_text())
    for i,row in enumerate(h):
        update=row['update'];(root/f'histories/{update}.json').write_text(json.dumps(h[:i+1]))
        envelope=msgpack.unpackb((old/f'snapshots/{update}.msgpack').read_bytes(),raw=False)
        envelope['provenance']=json.dumps(prov,sort_keys=True,separators=(',',':'));(root/f'snapshots/{update}.msgpack').write_bytes(msgpack.packb(envelope,use_bin_type=True))
    receipt=json.loads((old/'commit.json').read_text());receipt.update(checkpoint_sha256=digest(root/receipt['checkpoint']),history_sha256=digest(root/receipt['history']))
    contract=json.loads((old/'checkpoint-contract.json').read_text());contract.update(config_identity=prov['config_identity'],commit=receipt)
    for name,value in [('config.json',c),('case.json',c['cases'][0]),('provenance.json',prov),('checkpoint-contract.json',contract),('commit.json',receipt)]:
        (root/name).write_text(json.dumps(value))
    return root,s

def test_direct_probe_requires_full_audit_before_claim_or_engine(tmp_path,monkeypatch):
    replacement=module('width_rate_replacement');seen=[]
    def deny(config,root):seen.append('full_audit');raise ValueError('source-derived input mutation')
    monkeypatch.setattr(replacement,'audit_replacement_cell',deny)
    monkeypatch.setattr(replacement,'_engine',lambda:pytest.fail('numeric import before guard'))
    (tmp_path/'config.json').write_text(CONFIG.read_text())
    with pytest.raises(ValueError,match='source-derived'):replacement.replacement_probe(tmp_path,'a'*64)
    assert seen==['full_audit'] and not (tmp_path/'probe-entry.json').exists()

def test_tiny_nontrivial_provenance_fresh_restore(tmp_path):
    # Private numerical engineering route, not a production guard bypass.
    from test_width_rate_runs import fixture,tiny_probe
    from force_consistency import width_rate_runs as wr
    p,m,b,n,state,c,case,base=fixture()
    out=wr._run_arm(c,case,state,p,p,m,n,base,{},tmp_path/'tiny',{'historical':{200:{'value':2},1000:{'value':10}}},wr.ExecutionContext(120.).for_arm(120.,0.),probe_runner=tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    assert out['checkpoint_verification']['maximum_absolute_error']<=1e-10
    assert out['checkpoint_verification']['adam_count']==3
    assert set(out['provenance']['historical'])=={'200','1000'}

@pytest.mark.parametrize('damage',['template_frames','prediction_frames','history','tolerance','nonfinite'])
def test_strict_child_receipt_rejects_wrong_work_and_identity(tmp_path,damage):
    replacement=module('width_rate_replacement');root,sources=_staged_endpoint(tmp_path,replacement)
    audit=replacement._audit_cell(json.loads(CONFIG.read_text()),root,sources)
    child=json.loads(Path('runs/aspirin-width-rate-recovery-2026-10-01/summary.json').read_text())['child_receipt']
    child.update(config_identity='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36',checkpoint_sha256=audit['commit']['checkpoint_sha256'],history_sha256=audit['commit']['history_sha256'])
    assert replacement.valid_child(audit,child)['prediction_frames']==144
    if damage=='template_frames':child['template_initializer_frames']=15
    elif damage=='prediction_frames':child['prediction_frames']=143
    elif damage=='history':child['history_sha256']='0'*64
    elif damage=='tolerance':child['tolerance']=1e-8
    else:child['maximum_absolute_error']=float('nan')
    with pytest.raises(ValueError):replacement.valid_child(audit,child)

def test_direct_unreserved_public_probe_does_not_import_arrays(tmp_path):
    (tmp_path/'config.json').write_text(CONFIG.read_text());(tmp_path/'case.json').write_text(json.dumps(json.loads(CONFIG.read_text())['cases'][0]))
    code='''
import sys,importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('numpy','jax','flax','optax','e3x'):raise AssertionError('NUMERICAL_IMPORT')
sys.meta_path.insert(0,Deny())
from force_consistency.width_rate_replacement import replacement_probe
replacement_probe(sys.argv[1],'a'*64)
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Unreviewed replacement cell root' in result.stderr
    assert 'NUMERICAL_IMPORT' not in result.stderr
    assert not (tmp_path/'probe-entry.json').exists()

def test_controller_fallback_keeps_completed_and_failed_cells(tmp_path):
    replacement=module('width_rate_replacement');c=json.loads(CONFIG.read_text())
    first=c['cases'][0];second=c['cases'][1]
    for case,status,updates in ((first,'completed',6000),(second,'failed',200)):
        child=tmp_path/case['name'];child.mkdir();(child/'summary.json').write_text(json.dumps({'case':case,'status':status,'executed':True,'execution_valid':status=='completed','updates':updates}))
    out=replacement.failure_outcome(c,tmp_path,RuntimeError('controller failed'),[])
    assert out['arms'][first['name']]['status']=='completed'
    assert out['arms'][second['name']]['updates']==200
    assert [out['arms'][x['name']]['status'] for x in c['cases'][2:]]==['blocked','blocked']
    assert not out['execution_valid']

def test_launcher_rechecks_live_deadline_before_issuing_child(tmp_path,monkeypatch):
    replacement=module('width_rate_replacement')
    class Expired:
        def remaining_seconds(self):return -0.01
    monkeypatch.setattr(replacement.subprocess,'run',lambda *a,**k:pytest.fail('child issued after live deadline'))
    with pytest.raises(TimeoutError):replacement._launch(tmp_path,1,'a'*64,context=Expired())

def test_known_child_receipt_uses_cached_guard_after_later_source_failure(tmp_path,monkeypatch):
    replacement=module('width_rate_replacement');root,sources=_staged_endpoint(tmp_path,replacement)
    child=json.loads(Path('runs/aspirin-width-rate-recovery-2026-10-01/summary.json').read_text())['child_receipt']
    child.update(config_identity='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36',checkpoint_sha256=digest_for_test(root/'snapshots/6000.msgpack'),history_sha256=digest_for_test(root/'histories/6000.json'))
    (root/'checkpoint-verification.json').write_text(json.dumps(child))
    monkeypatch.setattr(replacement,'verify_replacement_sources',lambda c:(_ for _ in ()).throw(ValueError('later unrelated source failure')))
    known=replacement.recover_child_work(json.loads(CONFIG.read_text()),root,sources)
    assert known['work']['prediction_frames']==144
    assert known['receipt']['adam_count']==6000

def digest_for_test(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.mark.parametrize('damage',['earlier_update','wrong_count','wrong_norm','other_width'])
def test_serialized_payload_mutation_rejected_before_child(tmp_path,damage):
    import msgpack,hashlib
    replacement=module('width_rate_replacement');root,sources=_staged_endpoint(tmp_path,replacement)
    path=root/'snapshots/6000.msgpack';envelope=msgpack.unpackb(path.read_bytes(),raw=False)
    state=msgpack.unpackb(envelope['payload'],raw=False)
    if damage=='earlier_update':envelope['payload']=msgpack.unpackb((root/'snapshots/0.msgpack').read_bytes(),raw=False)['payload']
    else:
        if damage=='wrong_count':state['optimizer']['0']['count']=msgpack.ExtType(1,msgpack.packb([[],'int32',(5999).to_bytes(4,'little',signed=True)],use_bin_type=True))
        elif damage=='wrong_norm':state['normalization_json']='{}'
        else:
            def change(v):
                if isinstance(v,msgpack.ExtType) and v.code==1:
                    shape,dtype,raw=msgpack.unpackb(v.data,raw=False)
                    if shape and shape[-1]==16:return msgpack.ExtType(1,msgpack.packb([shape[:-1]+[32],dtype,raw+raw],use_bin_type=True))
                if isinstance(v,dict):return {k:change(x) for k,x in v.items()}
                return v
            state['params']=change(state['params'])
        envelope['payload']=msgpack.packb(state,use_bin_type=True)
    envelope['sha256']=hashlib.sha256(envelope['payload']).hexdigest();path.write_bytes(msgpack.packb(envelope,use_bin_type=True))
    receipt=json.loads((root/'commit.json').read_text());receipt['checkpoint_sha256']=digest_for_test(path);(root/'commit.json').write_text(json.dumps(receipt))
    contract=json.loads((root/'checkpoint-contract.json').read_text());contract['commit']=receipt;(root/'checkpoint-contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError):replacement._audit_cell(json.loads(CONFIG.read_text()),root,sources)
    assert not (root/'probe-attempt.json').exists()

def test_postclaim_root_replacement_leaves_unowned_files_untouched(tmp_path,monkeypatch):
    import time
    replacement=module('width_rate_replacement');root=tmp_path/'run';c=json.loads(CONFIG.read_text());c['run_directory']=str(root)
    class Context:
        def __init__(self,*a,**kw):self.global_start=kw.get('global_start',time.perf_counter());self.clock=time.perf_counter
        def check_budget(self):pass
    class Engine:
        ExecutionContext=Context
        @staticmethod
        def code_provenance(p):return {}
    monkeypatch.setattr(replacement,'verify_replacement_sources',lambda c:{'private_metadata_fixture':True})
    monkeypatch.setattr(replacement,'evidence',lambda *a:None)
    monkeypatch.setattr(replacement,'_engine',lambda:Engine)
    def race(config,context,partial):
        root.rename(tmp_path/'owned-original');root.mkdir();(root/'sources.json').write_text('UNOWNED')
        return {'sources':{},'widths':{},'parent_work':{}}
    monkeypatch.setattr(replacement,'replacement_preflight',race)
    out=replacement.diagnose_width_rate_replacement(c)
    assert (root/'sources.json').read_text()=='UNOWNED'
    assert not (root/'provenance.json').exists()
    assert not out['execution_valid']
    assert Path(out['external_failure_record']).exists()

def test_directory_handle_closes_check_to_replace_race(tmp_path,monkeypatch):
    replacement=module('width_rate_replacement');root=tmp_path/'owned';root.mkdir();owner=replacement.DirectoryLease(root);real=replacement.os.replace
    def race(src,dst,*a,**kw):
        root.rename(tmp_path/'original');root.mkdir();(root/'saved.json').write_text('UNOWNED')
        return real(src,dst,*a,**kw)
    monkeypatch.setattr(replacement.os,'replace',race)
    try:
        with pytest.raises(ValueError,match='ownership'):owner.json('saved.json',{'owned':True})
        assert (root/'saved.json').read_text()=='UNOWNED'
        assert json.loads((tmp_path/'original/saved.json').read_text())=={'owned':True}
    finally:owner.close()

def test_owned_numeric_engine_and_fresh_process_keep_case_paths(tmp_path):
    from test_width_rate_runs import fixture,tiny_probe
    from force_consistency import width_rate_runs as wr
    replacement=module('width_rate_replacement');repo=Path.cwd();root=tmp_path/'owned-case';root.mkdir()
    owner=replacement.DirectoryLease(root);p,m,b,n,state,c,case,base=fixture()
    context=replacement.OwnedContext(wr.ExecutionContext(120.).for_arm(120.,0.),owner)
    def probe(ignored,ctx):
        with replacement.repository_directory(repo):return tiny_probe(root,ctx)
    try:
        with owner.directory():out=wr._run_arm(c,case,state,p,p,m,n,base,{},Path('.'),{'owned_engine_fixture':True},context,probe_runner=probe)
        assert Path.cwd()==repo
        assert out['execution_valid'],out.get('failure_reason')
        assert json.loads((root/'commit.json').read_text())['checkpoint']=='snapshots/3.msgpack'
        assert out['checkpoint_verification']['adam_count']==3
    finally:owner.close()

def test_direct_payload_guard_denies_numerical_imports(tmp_path):
    replacement=module('width_rate_replacement');root,sources=_staged_endpoint(tmp_path,replacement)
    import msgpack,hashlib
    path=root/'snapshots/6000.msgpack';env=msgpack.unpackb(path.read_bytes(),raw=False)
    env['payload']=msgpack.unpackb((root/'snapshots/0.msgpack').read_bytes(),raw=False)['payload'];env['sha256']=hashlib.sha256(env['payload']).hexdigest();path.write_bytes(msgpack.packb(env,use_bin_type=True))
    receipt=json.loads((root/'commit.json').read_text());receipt['checkpoint_sha256']=digest_for_test(path);(root/'commit.json').write_text(json.dumps(receipt))
    contract=json.loads((root/'checkpoint-contract.json').read_text());contract['commit']=receipt;(root/'checkpoint-contract.json').write_text(json.dumps(contract))
    code='''
import sys,importlib.abc,json
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('jax','numpy','flax','e3x','optax'):raise AssertionError('NUMERICAL_IMPORT')
sys.meta_path.insert(0,Deny())
from force_consistency import width_rate_replacement as w
c=json.load(open('configs/aspirin-width-rate-replacement.json'));s=w.verify_replacement_sources(c)
w.audit_replacement_cell=lambda c,r:w._audit_cell(c,r,s) # private engineering root adapter; full state audit stays real
w.replacement_probe(sys.argv[1],'a'*64)
'''
    result=subprocess.run([sys.executable,'-c',code,str(root)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Serialized update/normalization changed' in result.stderr
    assert 'NUMERICAL_IMPORT' not in result.stderr
    assert not (root/'probe-entry.json').exists()
