import copy
import json
from pathlib import Path
import numpy as np
import pytest
from force_consistency import gradient_runs as runs
from force_consistency.diagnostics import sha256
from force_consistency.data.prepare import write_json

CONFIG=Path('configs/aspirin-representation-gradient-diagnostic.json')

@pytest.mark.parametrize('change',['omitted','wrong','setting'])
def test_source_gate_precedes_prepared_loader(monkeypatch,change):
    c=json.loads(CONFIG.read_text())
    if change=='omitted':c['source_sha256'].pop('diverse128/reference.npz')
    elif change=='wrong':c['source_sha256']['diverse128/reference.npz']='wrong'
    else:c['model']['features']=8
    def forbidden(*a,**k):raise AssertionError('prepared data accessed before source gate')
    monkeypatch.setattr(runs,'load_training_partition',forbidden)
    with pytest.raises(ValueError):runs.load_gradient_sources(c)


@pytest.fixture(scope='module')
def sources():
    return runs.load_gradient_sources(json.loads(CONFIG.read_text()))


def test_real_source_state_array_and_prediction_identity(sources):
    train,manifest,norm,subsets,fresh,final,historical=sources
    assert [len(subsets['subsets'][k]['record_ids']) for k in runs.ARMS]==[16,16,128]
    assert historical['verification']['source_predictions_max_absolute_difference']<1e-10
    assert all(int(s['update'])==2000 for s in final.values()) and int(fresh['update'])==0
    assert manifest['identity']==json.loads(CONFIG.read_text())['expected_prepared_identity']
    assert historical['verification']['initial_state_hashes']==runs._state_hashes(fresh)

@pytest.mark.parametrize('filename,mutation',[('narrow16/last.msgpack',None),('narrow16/config.json','setting'),('narrow16/fitting-inputs.npz','arrays')])
def test_rehashed_corrupt_source_rejected_before_diagnostics(tmp_path,filename,mutation):
    import shutil
    c=json.loads(CONFIG.read_text());root=tmp_path/'source';shutil.copytree(c['source_run'],root);c['source_run']=str(root)
    file=root/filename
    if mutation is None:file.write_bytes(file.read_bytes()[:-11]+b'badpayload!')
    elif mutation=='setting':
        d=json.loads(file.read_text());d['model']['features']=8;write_json(file,d)
    else:
        with np.load(file) as z:arrays={k:z[k].copy() for k in z.files}
        arrays['forces'][0,0,0]+=1.;np.savez(file,**arrays)
    c['source_sha256'][filename]=sha256(file)
    with pytest.raises((ValueError,TypeError)):runs.load_gradient_sources(c)


def case_fixture():
    from test_gradient_diagnostics import witness
    p,m,b,n,s=witness()
    return [(name,s,p) for name in runs.CASE_NAMES],m,n


def test_order_counts_no_state_update_and_partial_failure_retention(tmp_path):
    cases,m,n=case_fixture();before=[runs._state_hashes(s) for _,s,_ in cases]
    calls=[]
    def analyze(state,p,model,norm,config):
        name=runs.CASE_NAMES[len(calls)];calls.append(name)
        config['_save_partial']({'record_ids':list(p.record_ids),'observed':True})
        if len(calls)==2:raise RuntimeError('injected after retained partial')
        return {'verification_valid':True,'record_count':7,'record_ids':list(p.record_ids)}
    out=runs.execute_gradient_cases(tmp_path,cases,m,n,{},lambda:None,analyze)
    assert calls==list(runs.CASE_NAMES[:2])
    assert out[runs.CASE_NAMES[0]]['status']=='completed'
    assert out[runs.CASE_NAMES[1]]['status']=='failed' and out[runs.CASE_NAMES[1]]['executed']
    assert (tmp_path/runs.CASE_NAMES[1]/'partial.json').exists()
    assert all(out[k]['status']=='blocked' and not out[k]['executed'] for k in runs.CASE_NAMES[2:])
    assert before==[runs._state_hashes(s) for _,s,_ in cases]
    saved=json.loads((tmp_path/'cases.json').read_text());assert list(saved)==sorted(runs.CASE_NAMES)


def test_case_order_and_wall_blocks_without_attempt(tmp_path):
    cases,m,n=case_fixture();seen=[]
    def analyze(s,p,m,n,c):
        seen.append(runs.CASE_NAMES[len(seen)])
        return {'verification_valid':True,'record_count':7,'record_ids':list(p.record_ids)}
    out=runs.execute_gradient_cases(tmp_path/'all',cases,m,n,{},lambda:None,analyze)
    assert seen==list(runs.CASE_NAMES) and all(r['record_count']==7 for r in out.values())
    def expired():raise runs.WallBudget('cap')
    wall=runs.execute_gradient_cases(tmp_path/'wall',cases,m,n,{},expired,analyze)
    assert all(r['status']=='blocked' and not r['executed'] for r in wall.values())


def test_protected_output_and_preflight_failure_preserved(tmp_path):
    c=json.loads(CONFIG.read_text());c['run_directory']=str(tmp_path/'run');c.pop('evidence_log');c['source_sha256'].pop('config.json')
    out=runs.diagnose_gradients(c)
    assert not out['execution_valid'] and out['stage']=='preflight'
    assert all(v['status']=='blocked' for v in out['cases'].values())
    saved=(tmp_path/'run'/'summary.json').read_bytes()
    with pytest.raises(ValueError):runs.diagnose_gradients(c)
    assert (tmp_path/'run'/'summary.json').read_bytes()==saved


def test_cli_exit_uses_execution_validity(tmp_path,monkeypatch,capsys):
    from force_consistency.cli import main
    c=tmp_path/'c.json';write_json(c,{})
    monkeypatch.setattr(runs,'diagnose_gradients',lambda c:{'execution_valid':True,'learning_accepted':False,'status':'completed'})
    assert main(['diagnose-gradients','--config',str(c)])==0
    monkeypatch.setattr(runs,'diagnose_gradients',lambda c:{'execution_valid':False,'status':'failed'})
    assert main(['diagnose-gradients','--config',str(c)])==1
