"""Saved-only reconciliation; synthetic trials are engineering evidence only."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from force_consistency.width_rate_contract import digest
from test_width_rate_replacement import module, CONFIG
from test_width_rate_report import valid_arm


def generator():
    path=Path('scripts/report_width_rate_replacement.py');assert path.exists(),'Replacement report not implemented'
    spec=importlib.util.spec_from_file_location('replacement_report',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def arm_fixture(case=None):
    c=json.loads(CONFIG.read_text());s=module().verify_replacement_sources(c)
    _,_,a=valid_arm(case)
    if case is None:case=c['cases'][0]
    a['case']=case;a['provenance'].update(config_identity='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36',case=case,sources=json.loads(json.dumps(s)),source_pin_manifest_identity=c['source_pin_manifest_sha256'])
    a['checkpoint_verification']['config_identity']='d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36'
    return c,s,a

def test_explicit_new_validator_accepts_learning_failure():
    c,s,a=arm_fixture();m=generator();assert m.verified_arm(a,c,s)
    assert not a['fit_gate']['accepted']

@pytest.mark.parametrize('damage',['baseline','history','gate','config','child','state','oracle','shared','time','group','weighted','overlap'])
def test_new_validator_rejects_coherent_or_incomplete_claims(damage):
    c,s,a=arm_fixture();m=generator()
    if damage=='baseline':a['original_baselines']['fitting']['energy_mse_eV2']*=100
    elif damage=='history':a['history'].pop(5)
    elif damage=='gate':a['history'][-1]['fit_gate']['accepted']=True
    elif damage=='config':a['provenance']['config_identity']='e64165f0e511ae073dae1fb6e3da49053fee110e5844b5da0471bce47ec01198'
    elif damage=='child':a['checkpoint_verification']['template_initializer_frames']=15
    elif damage=='state':a['initial_state_hashes_after']['params']='0'*64
    elif damage=='oracle':a['oracle_verification']['fullbatch_reference_frames']=127
    elif damage=='shared':a['shared_initialization_seconds']=1000
    elif damage=='time':a['elapsed_seconds']=float('nan')
    elif damage=='group':a['history'][0]['metrics']['fitting']['per_group']={}
    elif damage=='weighted':a['history'][2]['weighted_loss']['fitting']*=2
    else:a['history'][0]['metrics']['anchor_fitted_flags']={}
    assert not m.verified_arm(a,c,s)

@pytest.mark.parametrize('terminal',['missing','corrupt','nonfinite','fake_failed'])
def test_partial_report_preserves_four_statuses_and_unknown_work(tmp_path,terminal,monkeypatch):
    m=generator();c=json.loads(CONFIG.read_text());root=tmp_path/'run';root.mkdir();out=tmp_path/'reports'
    (root/'config.json').write_text(CONFIG.read_text())
    if terminal=='corrupt':(root/'summary.json').write_text('{broken')
    elif terminal=='nonfinite':(root/'summary.json').write_text('{"status":"failed","elapsed_seconds":NaN}')
    elif terminal=='fake_failed':
        (root/'summary.json').write_text(json.dumps({'status':'failed','arms':{c['cases'][0]['name']:{'case':c['cases'][0],'status':'failed','executed':True,'updates':999999,'attempted_updates':999999,'persisted_update':999999,'elapsed_seconds':-3}}}))
    monkeypatch.setattr(m,'_figure',lambda *a:None)
    result=m.generate(root,out)
    assert not result['interpretation_valid'] and not result['contrasts']['available']
    assert set(result['arms'])=={case['name'] for case in c['cases']}
    assert result['actual_counts']['committed_updates']==0
    assert result['actual_counts']['synchronized_updates'] is None
    assert result['actual_matrix_elapsed_seconds'] is None
    assert (out/'width-rate-replacement-diagnostics.json').exists()

@pytest.mark.parametrize('missing',['deferred_review','preservation','navigation','results_commit','negative_case'])
def test_final_completeness_requires_late_artifacts_and_all_cases(tmp_path,missing):
    m=generator();evidence=tmp_path/'evidence.md';cases={'one':{'status':'failed'},'two':{'status':'blocked'}}
    evidence.write_text('\n'.join('CASE:'+name+':'+arm['status'] for name,arm in cases.items()))
    roles={name:tmp_path/(name+'.txt') for name in ('deferred_review','preservation','navigation','results_commit','code','config','pins','raw','reports','tests','plan_review','engineering_logs','ancestry')}
    for p in roles.values():p.write_text('durable')
    artifacts={'stage':'final','evidence_prefix_bytes':len(evidence.read_bytes()),'evidence_prefix_sha256':digest(evidence),
        'required_artifacts':{str(p):digest(p) for p in roles.values()},'required_roles':{name:str(p) for name,p in roles.items()},
        'case_evidence':{name:'CASE:'+name+':'+arm['status'] for name,arm in cases.items()},'required_evidence':['CASE:one:failed','CASE:two:blocked'],
        'exclusions':['coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha']}
    assert not m.reconcile_evidence({'arms':cases},evidence,artifacts)['log_completeness']
    if missing=='negative_case':artifacts['case_evidence'].pop('two')
    else:artifacts['required_artifacts'].pop(str(roles[missing]))
    assert not m.reconcile_evidence({'arms':cases},evidence,artifacts)['log_completeness']

def test_unstructured_failed_blocked_markers_never_certify_closure(tmp_path):
    m=generator();evidence=tmp_path/'e.md';evidence.write_text('CASE:a:failed\nCASE:b:blocked\ntrial and tests')
    roles={name:tmp_path/(name+'.txt') for name in ('deferred_review','preservation','navigation','results_commit','code','config','pins','raw','reports','tests','plan_review','engineering_logs','ancestry')}
    for p in roles.values():p.write_text('durable')
    artifacts={'stage':'final','evidence_prefix_bytes':len(evidence.read_bytes()),'evidence_prefix_sha256':digest(evidence),
        'required_artifacts':{str(p):digest(p) for p in roles.values()},'required_roles':{name:str(p) for name,p in roles.items()},
        'case_evidence':{'a':'CASE:a:failed','b':'CASE:b:blocked'},'required_evidence':['trial and tests'],
        'exclusions':['coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha']}
    result={'arms':{'a':{'status':'failed'},'b':{'status':'blocked'}}}
    assert not m.reconcile_evidence(result,evidence,artifacts)['log_completeness']
    with evidence.open('a') as f:f.write('\nexplicit closing entry')
    assert not m.reconcile_evidence(result,evidence,artifacts)['log_completeness']
    roles['deferred_review'].write_text('changed')
    assert not m.reconcile_evidence(result,evidence,artifacts)['log_completeness']

def test_preliminary_never_claims_final_completeness(tmp_path):
    m=generator();evidence=tmp_path/'e.md';evidence.write_text('pending')
    assert not m.reconcile_evidence({'arms':{}},evidence,{'stage':'preliminary'})['log_completeness']


def test_four_replacement_endpoints_allow_descriptive_contrasts():
    m=generator();c=json.loads(CONFIG.read_text());arms={}
    for case in c['cases']:
        _,sources,arm=arm_fixture(case)
        assert m.verified_arm(arm,c,sources)
        arms[case['name']]={**arm,'interpretation_valid':True}
    assert m.factorial_contrasts(arms)['available']
    arms[c['cases'][1]['name']]['execution_valid']=False
    assert not m.factorial_contrasts(arms)['available']

def test_report_can_run_with_engine_imports_and_array_loading_denied(tmp_path):
    import subprocess,sys
    root=tmp_path/'raw';root.mkdir();(root/'config.json').write_text(CONFIG.read_text())
    code="""
import importlib.abc,importlib.util,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('jax','flax','optax','e3x'):raise AssertionError('ENGINE_IMPORT')
sys.meta_path.insert(0,Deny())
import numpy
numpy.load=lambda *a,**k: (_ for _ in ()).throw(AssertionError('ARRAY_LOAD'))
spec=importlib.util.spec_from_file_location('report','scripts/report_width_rate_replacement.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m._figure=lambda *a:None
m.generate(sys.argv[1],sys.argv[2])
"""
    result=subprocess.run([sys.executable,'-c',code,str(root),str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'out/width-rate-replacement-diagnostics.json').exists()

def test_completeness_requires_code_config_raw_tests_and_ancestry_roles(tmp_path):
    m=generator();e=tmp_path/'e.md';e.write_text('CASE:a:failed')
    roles={name:tmp_path/(name+'.txt') for name in ('deferred_review','preservation','navigation','results_commit')}
    for p in roles.values():p.write_text('available')
    manifest={'stage':'final','evidence_prefix_bytes':len(e.read_bytes()),'evidence_prefix_sha256':digest(e),
        'required_artifacts':{str(p):digest(p) for p in roles.values()},'required_roles':{r:str(p) for r,p in roles.items()},
        'case_evidence':{'a':'CASE:a:failed'},'required_evidence':[],
        'exclusions':['coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha']}
    assert not m.reconcile_evidence({'arms':{'a':{'status':'failed'}}},e,manifest)['log_completeness']

def test_failed_elapsed_requires_a_bound_terminal_publication_receipt(tmp_path,monkeypatch):
    m=generator();root=tmp_path/'raw';root.mkdir();(root/'config.json').write_text(CONFIG.read_text())
    (root/'summary.json').write_text(json.dumps({'status':'failed','elapsed_seconds':42.,'arms':{}}))
    monkeypatch.setattr(m,'_figure',lambda *a:None)
    result=m.generate(root,tmp_path/'out')
    assert result['actual_matrix_elapsed_seconds'] is None
    assert result['raw_terminal_elapsed_seconds']==42.

@pytest.mark.parametrize('damage',['summary_list','null_arms','missing_status','scalar_events','source_guard'])
def test_malformed_structures_produce_complete_incomplete_outputs(tmp_path,monkeypatch,damage):
    m=generator();root=tmp_path/'raw';root.mkdir();(root/'config.json').write_text(CONFIG.read_text())
    summary={'status':'failed','arms':{}}
    if damage=='summary_list':summary=[1]
    elif damage=='null_arms':summary['arms']=None
    elif damage=='missing_status':summary['arms']={'width16-lr1e-3':{'executed':True}}
    elif damage=='scalar_events':(root/'preflight-work.json').write_text('[1]')
    else:monkeypatch.setattr(m,'verify_replacement_sources',lambda c:(_ for _ in ()).throw(ValueError('source guard failed')))
    (root/'summary.json').write_text(json.dumps(summary));monkeypatch.setattr(m,'_figure',lambda *a:None)
    result=m.generate(root,tmp_path/'out')
    assert not result['interpretation_valid'] and len(result['arms'])==4
    assert (tmp_path/'out/width-rate-replacement-diagnostics.md').exists()
    assert result['verification_errors']

def test_fabricated_preflight_and_blocked_counts_stay_unknown(tmp_path,monkeypatch):
    m=generator();root=tmp_path/'raw';root.mkdir();(root/'config.json').write_text(CONFIG.read_text());c=json.loads(CONFIG.read_text())
    (root/'summary.json').write_text(json.dumps({'status':'failed','arms':{x['name']:{'status':'blocked','executed':False} for x in c['cases']}}))
    (root/'preflight-work.json').write_text(json.dumps([{'operation':'shared_training_initialization','features':16,'status':'completed','completed_initializations':999999}]))
    monkeypatch.setattr(m,'_figure',lambda *a:None);result=m.generate(root,tmp_path/'out')
    assert result['actual_counts']['shared_training_initializations_completed'] is None
    assert result['actual_counts']['synchronized_updates'] is None
    assert len(result['actual_counts']['unknown_probe_cells'])==4

def test_plan_markers_and_lock_cannot_pose_as_final_evidence(tmp_path):
    m=generator();c=json.loads(CONFIG.read_text());s=module().verify_replacement_sources(c);e=tmp_path/'e.md';e.write_text('WIDTHRATE-REPLACEMENT-PLAN-001 prospective only')
    result={'arms':{x['name']:{'case':x,'status':'failed' if i==0 else 'blocked'} for i,x in enumerate(c['cases'])},'sources':s,'source_root':str(tmp_path/'absent'),'artifacts_sha256':{}}
    roles=('deferred_review','preservation','navigation','results_commit','code','config','pins','raw','reports','tests','plan_review','engineering_logs','ancestry')
    manifest={'stage':'final','evidence_prefix_bytes':len(e.read_bytes()),'evidence_prefix_sha256':digest(e),'required_artifacts':s['pins'],
        'required_roles':{r:'uv.lock' for r in roles},'case_evidence':{n:'WIDTHRATE-REPLACEMENT-PLAN-001' for n in result['arms']},'required_evidence':[],
        'exclusions':['coverage_outputs','coverage_manifest','evidence_after_cutoff','closing_commit_sha']}
    assert not m.reconcile_evidence(result,e,manifest)['log_completeness']


def test_structured_negative_outcome_binds_status_counts_and_artifact_links():
    m=generator();arm={'case':{'name':'failed'},'status':'failed','execution_valid':False,'interpretation_valid':False,
        'stop_reason':'wall_budget','persisted_update':200,'known_child_work':None,
        'derived_synchronized_updates':None,'derived_attempted_updates':None,
        'raw_terminal_claims':{'updates':203,'attempted_updates':204,'failure_reason':'timeout'}}
    files={'failed/commit.json':'a'*64,'blocked/config.json':'b'*64}
    record=m.case_record('failed',arm,files)
    assert record['raw_updates']==203 and record['raw_attempted_updates']==204 and record['persisted_update']==200
    assert record['known_child_work'] is None and record['derived_attempted_updates'] is None
    assert record['raw_files_sha256']=={'failed/commit.json':'a'*64}
    text='## CASE-001 — bound outcome\n\n```json\n'+json.dumps(record)+'\n```\n'
    assert m.structured_evidence(text,'CASE-001')==record
    changed=json.loads(json.dumps(arm));changed['raw_terminal_claims']['updates']=999
    assert m.structured_evidence(text,'CASE-001')!=m.case_record('failed',changed,files)
    with pytest.raises(ValueError):m.structured_evidence('## CASE-001 — planned only','CASE-001')

def test_blocked_navigation_requires_explicit_typed_reason(tmp_path):
    m=generator();path=tmp_path/'navigation.json'
    value={'schema':'replacement-navigation-v1','scope':'width-rate-replacement-2026-10-01',
        'config_identity':'d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36',
        'status':'blocked','reason':'host CLI command denied','attempted_command':'obsidian vault=hermes read'}
    path.write_text(json.dumps(value));assert m.artifact_role_valid('navigation',str(path),{},{} )
    value.pop('reason');path.write_text(json.dumps(value));assert not m.artifact_role_valid('navigation',str(path),{},{} )

def test_substituted_partial_payload_does_not_promote_persisted_counts(tmp_path,monkeypatch):
    import msgpack,hashlib
    from test_width_rate_replacement import _staged_endpoint
    replacement=module('width_rate_replacement');child,s=_staged_endpoint(tmp_path,replacement)
    c=json.loads(CONFIG.read_text());root=tmp_path/'matrix';root.mkdir();child.rename(root/c['cases'][0]['name']);child=root/c['cases'][0]['name']
    (root/'config.json').write_text(CONFIG.read_text())
    env=msgpack.unpackb((child/'snapshots/6000.msgpack').read_bytes(),raw=False)
    env['payload']=msgpack.unpackb((child/'snapshots/0.msgpack').read_bytes(),raw=False)['payload'];env['sha256']=hashlib.sha256(env['payload']).hexdigest();(child/'snapshots/6000.msgpack').write_bytes(msgpack.packb(env,use_bin_type=True))
    receipt=json.loads((child/'commit.json').read_text());receipt['checkpoint_sha256']=digest(child/'snapshots/6000.msgpack');(child/'commit.json').write_text(json.dumps(receipt))
    (root/'summary.json').write_text(json.dumps({'status':'failed','arms':{c['cases'][0]['name']:{'case':c['cases'][0],'status':'failed','executed':True,'updates':6000}}}))
    m=generator();monkeypatch.setattr(m,'_figure',lambda *a:None);result=m.generate(root,tmp_path/'out')
    assert result['actual_counts']['committed_updates'] is None
    assert c['cases'][0]['name'] in result['actual_counts']['unknown_committed_cells']

@pytest.mark.parametrize('payload',[[],None,{'scope':'width-rate-replacement-2026-10-01','config_identity':'d0b1637db732b7bccc8a9828b6e17843218e3f1a22cfa0d47e928ff9368b7f36','files_sha256':[]}])
def test_malformed_late_artifact_is_rejected_without_crash(tmp_path,payload):
    m=generator();path=tmp_path/'role.json';path.write_text(json.dumps(payload))
    assert not m.artifact_role_valid('code',str(path),{},{} )
