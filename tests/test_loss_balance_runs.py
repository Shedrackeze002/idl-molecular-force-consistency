import copy
import json
from pathlib import Path
import pytest
from force_consistency import loss_balance_runs as runs
from force_consistency.data.prepare import write_json
from test_coverage import seven_case,arm_config
from force_consistency.coverage_runs import run_coverage_arm

CONFIG=Path('configs/aspirin-loss-balance-diagnostic.json')

@pytest.mark.parametrize('mutate',['source','calibration','weight','ratio'])
def test_all_identity_and_calibration_gates_precede_loader(monkeypatch,mutate):
    c=json.loads(CONFIG.read_text())
    if mutate=='source':c['source_sha256'].pop('diverse128/history.json')
    elif mutate=='calibration':c['calibration_sha256']='wrong'
    elif mutate=='weight':c['arms']['balanced']['energy']=.2
    else:c['calibration_ratio']=.3
    def forbidden(*a,**k):raise AssertionError('Data opened before fixed calibration check')
    monkeypatch.setattr(runs,'load_gradient_sources',forbidden)
    with pytest.raises(ValueError):runs.balance_preflight(c)

@pytest.fixture(scope='module')
def preflight():return runs.balance_preflight(json.loads(CONFIG.read_text()))


def test_frozen_calibration_and_raw_baselines(preflight):
    train,manifest,norm,selected,historical,calibration=preflight
    assert calibration['raw']['record_count']==128 and calibration['raw']['force_component_count']==8064
    assert calibration['raw']['record_ids']==selected['subsets']['diverse128']['record_ids']
    assert calibration['ratio']==136.13314964807924
    assert historical['verification']['initial_state_hashes']==calibration['raw']['state_hashes_before']
    assert len(historical['arms']['diverse128']['summary']['history'])==22


def test_fitting_initial_guard_prevents_first_update(tmp_path):
    p,m,b,n,s=seven_case();c=arm_config(n)
    result=run_coverage_arm(c,p,p,tmp_path/'bad',{'fitting_initial_mse':{'energy_mse_eV2':123.,'force_mse_eV2_per_A2':123.}})
    assert not result['execution_valid'] and result['updates']==0
    assert 'fitting baseline' in result['failure_reason'].lower()


def history_case():
    source=json.loads(Path('runs/aspirin-data-coverage-2026-09-30/diverse128/summary.json').read_text())
    return source,copy.deepcopy(source)


def test_replay_compares_both_metrics_partitions_all22_and_duplicates():
    source,current=history_case();assert runs.verify_balance_replay(current,source,{'atol':1e-10,'rtol':1e-8})['compared_updates']==[r['update'] for r in source['history']]
    for role in ('fitting','anchor'):
        for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
            bad=copy.deepcopy(current);bad['history'][3]['metrics'][role][key]+=.01
            with pytest.raises(ValueError):runs.verify_balance_replay(bad,source,{'atol':1e-10,'rtol':1e-8})
    for history in [current['history'][:-1],[*current['history'],current['history'][-1]]]:
        bad={**current,'history':history}
        with pytest.raises(ValueError):runs.verify_balance_replay(bad,source,{'atol':1e-10,'rtol':1e-8})


def test_replay_comparison_failure_retains_completed_arm_and_blocks(tmp_path):
    def runner(name,expected):
        path=tmp_path/name;path.mkdir();write_json(path/'history.json',[{'update':2}]);write_json(path/'summary.json',{'status':'completed','updates':2})
        return {'status':'completed','executed':True,'execution_valid':True,'updates':2,'initial_state_hashes':{'params':'fresh','optimizer':'fresh'},'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    def mismatch(arm):raise ValueError('deliberate replay mismatch')
    result=runs.execute_balance_matrix(tmp_path,runner,mismatch)
    assert not result['execution_valid'] and result['next_step']=='resolve_execution_failure'
    assert result['arms']['replay']['status']=='completed'
    assert json.loads((tmp_path/'replay/summary.json').read_text())['status']=='completed'
    assert (tmp_path/'replay/replay-comparison-failure.json').exists()
    assert all(result['arms'][name]['status']=='blocked' for name in ('balanced','force-only'))


def test_uncaught_partial_runner_failure_retains_artifacts(tmp_path):
    def runner(name,expected):
        path=tmp_path/name;path.mkdir();write_json(path/'history.json',[{'update':5}]);(path/'last.msgpack').write_bytes(b'fixture')
        raise RuntimeError('after checkpoint')
    result=runs.execute_balance_matrix(tmp_path,runner,lambda r:{'status':'passed'})
    assert result['arms']['replay']['executed'] and result['arms']['replay']['updates']==5
    assert (tmp_path/'replay/last.msgpack').read_bytes()==b'fixture'
    assert all(result['arms'][name]['status']=='blocked' for name in ('balanced','force-only'))


def test_learning_failure_continues_and_force_only_not_joint_success(tmp_path):
    called=[]
    def runner(name,expected):
        called.append(name);(tmp_path/name).mkdir()
        return {'status':'completed','executed':True,'execution_valid':True,'updates':2000,'initial_state_hashes':{'params':'same','optimizer':'same'},'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    result=runs.execute_balance_matrix(tmp_path,runner,lambda r:{'status':'passed'})
    assert called==['replay','balanced','force-only'] and result['execution_valid']
    assert not result['balanced_joint_accepted'] and result['full_pilot']=='pending'
    with pytest.raises(ValueError):runs.execute_balance_matrix(tmp_path,runner,lambda r:{'status':'passed'})


def test_preflight_failure_blocks_and_protected_directory(tmp_path):
    c=json.loads(CONFIG.read_text());c['run_directory']=str(tmp_path/'run');c.pop('evidence_log');c['calibration_sha256']='bad'
    result=runs.diagnose_loss_balance(c)
    assert not result['execution_valid'] and all(not r['executed'] for r in result['arms'].values())
    saved=(tmp_path/'run/summary.json').read_bytes()
    with pytest.raises(ValueError):runs.diagnose_loss_balance(c)
    assert (tmp_path/'run/summary.json').read_bytes()==saved


def test_cli_uses_operational_success_not_joint_gate(tmp_path,monkeypatch):
    from force_consistency.cli import main
    path=tmp_path/'config.json';write_json(path,{})
    monkeypatch.setattr(runs,'diagnose_loss_balance',lambda c:{'execution_valid':True,'balanced_joint_accepted':False,'status':'completed'})
    assert main(['diagnose-loss-balance','--config',str(path)])==0


def test_raw_calibration_is_rechecked_after_each_arm(monkeypatch,tmp_path):
    c=json.loads(CONFIG.read_text());compact=json.loads(Path(c['calibration_path']).read_text())
    original=runs.sha256
    def changed(path):
        if str(path)==str(Path(c['calibration_run'])/'fresh-C-fitting/observations.json'):return 'changed'
        return original(path)
    monkeypatch.setattr(runs,'sha256',changed)
    with pytest.raises(ValueError,match='Calibration raw identity'):
        runs.verify_calibration_hashes(c)


@pytest.mark.parametrize('change',['count','update','ids','state'])
def test_calibration_metadata_rejected_even_with_matching_file_hashes(tmp_path,monkeypatch,change):
    from force_consistency.diagnostics import sha256
    c=json.loads(CONFIG.read_text());source=Path(c['calibration_run']);root=tmp_path/'calibration';root.mkdir()
    compact=json.loads(Path(c['calibration_path']).read_text())
    for name in compact['sources']:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((source/name).read_bytes())
    raw=json.loads((root/'fresh-C-fitting/observations.json').read_text())
    if change=='count':raw['record_count']=127;compact['cases']['fresh-C-fitting']['record_count']=127
    elif change=='update':raw['update']=1;compact['cases']['fresh-C-fitting']['update']=1
    elif change=='ids':raw['record_ids']=list(reversed(raw['record_ids']))
    else:
        raw['state_hashes_after']={'params':'different','optimizer':'different'}
        compact['cases']['fresh-C-fitting']['state_hashes_after']=raw['state_hashes_after']
    write_json(root/'fresh-C-fitting/observations.json',raw)
    compact['sources']['fresh-C-fitting/observations.json']=sha256(root/'fresh-C-fitting/observations.json')
    path=tmp_path/'compact.json';write_json(path,compact);c.update(calibration_run=str(root),calibration_path=str(path),calibration_sha256=sha256(path))
    def forbidden(*a,**k):raise AssertionError('Data opened before calibration metadata check')
    monkeypatch.setattr(runs,'load_gradient_sources',forbidden)
    with pytest.raises(ValueError):runs.balance_preflight(c)
