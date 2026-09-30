"""Trust-boundary regressions: no coherently edited controls reach numerical work."""
import copy
import json
from pathlib import Path
import pytest


def config():return json.loads(Path('configs/aspirin-width-rate-diagnostic.json').read_text())

def histories():
    return [json.loads(Path(f'runs/{name}/balanced/history.json').read_text()) for name in ('aspirin-loss-balance-2026-09-30','aspirin-convergence-budget-2026-09-30')]

@pytest.mark.parametrize('change',['weights','width','baseline','extra','pins'])
def test_contract_coherent_rebase_before_arrays(monkeypatch,change):
    from force_consistency import width_rate_runs as wr
    c=config()
    if change=='weights':c['loss_weights']['energy']*=2
    elif change=='width':c['cases'][0]['features']=32
    elif change=='baseline':c['gate_fraction']=.2
    elif change=='extra':c['allow_rebase']=True
    else:c['source_pin_manifest_sha256']='0'*64
    monkeypatch.setattr(wr,'balance_preflight',lambda *a:pytest.fail('arrays reached before trust guard'))
    with pytest.raises(ValueError):wr.width_rate_preflight(c,wr.ExecutionContext(11400),lambda d:None)

@pytest.mark.parametrize('damage',['replacement','missing'])
def test_pin_replacement_and_missing_file_before_arrays(tmp_path,monkeypatch,damage):
    from force_consistency import width_rate_runs as wr
    c=config();manifest=json.loads(Path(c['source_pin_manifest']).read_text())
    if damage=='replacement':manifest['files_sha256'].pop(next(iter(manifest['files_sha256'])))
    # Test malformed/missing original-path manifest without copying any numerical sources.
    monkeypatch.chdir(tmp_path);p=Path(c['source_pin_manifest']);p.parent.mkdir()
    if damage=='replacement':p.write_text(json.dumps(manifest))
    monkeypatch.setattr(wr,'balance_preflight',lambda *a:pytest.fail('arrays reached'))
    with pytest.raises((ValueError,FileNotFoundError)):wr.width_rate_preflight(c,wr.ExecutionContext(11400),lambda d:None)


def test_historical_duplicate_2000_must_match():
    from force_consistency.width_rate_contract import join_histories
    a,b=histories();b[0]['metrics']['fitting']['force_mse_eV2_per_A2']*=2
    with pytest.raises(ValueError):join_histories(a,b)


def test_historical_schedule_contains_250_not_1():
    from force_consistency.width_rate_contract import join_histories,verify_width_rate_sources
    a,b=histories();joined=join_histories(a,b)
    assert len(joined)==63 and 250 in joined and 2001 in joined and 1 not in joined
    result=verify_width_rate_sources(config())
    assert result['historical']==joined and result['pin_count']==326
    assert result['original_baselines']==a[0]['metrics']
    for damaged in (a[:-1],a[:2]+a[3:],a+[copy.deepcopy(a[0])]):
        with pytest.raises(ValueError):join_histories(damaged,b)

@pytest.mark.parametrize('kind',['before','after','interrupt'])
def test_parent_deadline_and_unknown_work(monkeypatch,kind):
    from force_consistency import width_rate_runs as wr
    now=[0.];context=wr.ExecutionContext(2.,clock=lambda:now[0]);records=[]
    def parent(c):
        now[0]=3.
        if kind=='interrupt':raise KeyboardInterrupt('pending parent work')
        return ('prepared',)
    monkeypatch.setattr(wr,'balance_preflight',parent)
    if kind=='before':now[0]=3.
    with pytest.raises((wr.WallBudget,KeyboardInterrupt)):
        wr.parent_operation({},context,lambda r:records.append(copy.deepcopy(r)))
    if kind=='before':assert records==[]
    elif kind=='after':
        assert records[-1]['completed_source_prediction_frames']==208
        assert records[-1]['historical_template_initializations']==1
    else:
        assert records[-1]['status']=='unknown' and records[-1]['completed_source_prediction_frames'] is None
        assert records[-1]['historical_template_initializations'] is None


def test_arm_context_uses_smaller_remaining_charge_and_global_allowance():
    from force_consistency import width_rate_runs as wr
    now=[0.];global_context=wr.ExecutionContext(10.,clock=lambda:now[0]);now[0]=2.
    arm=global_context.for_arm(6.,shared_initialization_seconds=1.)
    assert arm.remaining_seconds()==5.
    now[0]=7.
    with pytest.raises(wr.WallBudget):arm.check_budget()
    now[0]=2.;arm=global_context.for_arm(20.,shared_initialization_seconds=1.)
    assert arm.remaining_seconds()==8.


def test_pair_initial_hashes_and_zero_moments_and_original_baselines():
    from test_coverage import seven_case
    from force_consistency import width_rate_runs as wr
    from force_consistency.coverage_runs import coverage_metrics,_state_hashes
    import jax
    import numpy as np
    p,m,b,n,s=seven_case();base=coverage_metrics(s['params'],p,p,m,3);records=[]
    result=wr.initialize_width_states(config(),p,p,n,base,wr.ExecutionContext(120.),records.append)
    assert set(result)=={16,32}
    assert [r['status'] for r in records]==['started','completed','started','completed']
    assert sum(r['completed_initializations'] or 0 for r in records)==2
    for width,entry in result.items():
        state=entry['state'];wr.validate_state(state,n,0)
        assert int(state['optimizer'][0].count)==0
        assert all(np.count_nonzero(x)==0 for x in jax.tree.leaves((state['optimizer'][0].mu,state['optimizer'][0].nu)))
        assert _state_hashes(copy.deepcopy(state))==_state_hashes(state)
        assert entry['model'].config.features==width
        assert entry['model'].config.iterations==2
        assert entry['parameter_count']>0
        for role in ('fitting','anchor'):
            for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
                assert entry['initial_metrics'][role][key]==pytest.approx(base[role][key],abs=1e-10)
    assert result[32]['parameter_count']>result[16]['parameter_count']
    assert _state_hashes(result[16]['state'])!=_state_hashes(result[32]['state'])
    bad=copy.deepcopy(result[16]['state']);bad['update']=1
    with pytest.raises(ValueError):wr.validate_state(bad,n,0)
    bad=copy.deepcopy(result[16]['state']);bad['normalization_json']='{}'
    with pytest.raises(ValueError):wr.validate_state(bad,n,0)


def test_real_preflight_uses_only_training_arrays_and_frozen_support(monkeypatch):
    from force_consistency import width_rate_runs as wr
    import numpy as np
    original=np.load
    def guarded(path,*a,**kw):
        assert 'validation' not in str(path) and 'test.npz' not in str(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',guarded);records=[]
    result=wr.width_rate_preflight(config(),wr.ExecutionContext(120.),records.append)
    assert (len(result['fitting'].record_ids),len(result['anchor'].record_ids))==(128,16)
    assert set(result['anchor'].record_ids)<=set(result['fitting'].record_ids)
    assert result['parent_work']['completed_source_prediction_frames']==208
    assert result['sources']['pin_count']==326
    assert set(result['widths'])=={16,32}
