"""Protect residual science, immutable inputs and bounded failure accounting."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/diagnose_capacity_spikes.py'


def api():
    spec = importlib.util.spec_from_file_location('capacity_spikes', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('errors,mean2,variance,share', [
    ([2., 2.], 4., 0., 1.), ([-1., 1.], 0., 1., 0.),
    ([1., 3.], 4., 1., .8), ([0., 0.], 0., 0., None)])
def test_offset_and_variance_are_distinct_raw_mse_components(errors, mean2, variance, share):
    result = api().residual_summary(np.array(errors), np.zeros((2, 1, 3)), ['a', 'a'])
    assert result['mean_energy_error_squared_eV2'] == mean2
    assert result['centered_energy_error_variance_eV2'] == variance
    assert result['energy_mse_eV2'] == mean2 + variance
    assert result['offset_share'] == share
    assert result['offset_share_unavailable_reason'] == ('zero_energy_mse' if share is None else None)


def test_global_variance_keeps_between_group_means_and_unequal_group_weighting():
    result = api().residual_summary(np.array([0., 0., 3.]), np.zeros((3, 1, 3)), ['a', 'a', 'b'])
    assert result['energy_mse_eV2'] == 3.
    assert result['centered_energy_error_variance_eV2'] == 2.
    assert result['equal_group']['energy_mse_eV2'] == 4.5
    assert result['per_group']['a']['centered_energy_error_variance_eV2'] == 0.
    assert result['per_group']['b']['centered_energy_error_variance_eV2'] == 0.


@pytest.mark.parametrize('errors,forces,groups', [
    ([float('nan')], np.zeros((1, 1, 3)), ['a']),
    ([1., 2.], np.zeros((1, 1, 3)), ['a', 'a']),
    ([], np.zeros((0, 1, 3)), []),
    ([1.], np.zeros((1, 1, 3)), [])])
def test_invalid_residual_inputs_rejected(errors, forces, groups):
    with pytest.raises(ValueError):
        api().residual_summary(np.array(errors), forces, groups)


def test_inventory_rejects_omissions_extras_and_tampered_bytes(tmp_path):
    m = api(); root = tmp_path / 'source'; root.mkdir()
    (root / 'a').write_bytes(b'original'); pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps({'a': m.digest(root / 'a')})); identity = m.digest(pins)
    assert m.verify_inventory(root, pins, identity) == 1
    (root / 'b').write_bytes(b'extra')
    with pytest.raises(ValueError, match='members'): m.verify_inventory(root, pins, identity)
    (root / 'b').unlink(); (root / 'a').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'): m.verify_inventory(root, pins, identity)
    pins.write_text('{}')
    with pytest.raises(ValueError, match='identity'): m.verify_inventory(root, pins, identity)


def test_increment_shares_stay_signed_and_can_exceed_one():
    m = api()
    a = m.residual_summary(np.array([-1., 1.]), np.zeros((2, 1, 3)), ['a', 'a'])
    b = m.residual_summary(np.array([2., 2.]), np.zeros((2, 1, 3)), ['a', 'a'])
    d = m.residual_change(a, b)
    assert d['energy_mse_change_eV2'] == 3.
    assert d['mean_squared_change_eV2'] == 4.
    assert d['variance_change_eV2'] == -1.
    assert d['offset_increase_share'] == pytest.approx(4 / 3)
    assert m.residual_change(b, a)['offset_increase_share'] is None


def test_bias_derivative_counts_atoms_and_preserves_error_sign():
    m = api()
    check = m.bias_check(np.array([1., 3.]), np.array([21, 21]), 2., 42., 0., .25)
    assert check['expected_unweighted_energy'] == 21.
    assert not check['valid']  # A per-molecule or wrong-scale derivative must not pass.
    assert m.bias_check(np.array([1., 3.]), np.array([21, 21]), 2., 21., 0., .25)['valid']
    assert m.bias_check(np.array([-1., -3.]), np.array([21, 21]), 2., -21., 0., .25)['expected_weighted_energy'] == -5.25


def test_weighted_slopes_have_explicit_objective_and_direction():
    m = api(); ge = {'x': np.array([1., 0.])}; gf = {'x': np.array([-2., 1.])}
    # Weighted joint = -5,3. Dot products give E slope +5 and F slope -13.
    gj = {'x': np.array([-1., 1.])}; gw = {'x': np.array([-5., 3.])}
    r = m.gradient_metrics(ge, gf, gj, gw, {'energy': 1., 'force': 3.})
    whole = r['unweighted']['whole']
    assert whole['energy_scaled_loss_negative_weighted_joint_slope'] == 5.
    assert whole['force_scaled_loss_negative_weighted_joint_slope'] == pytest.approx(-13.)
    assert r['weighted']['whole']['force_weighted_component_negative_weighted_joint_slope'] == pytest.approx(-39.)
    assert r['unweighted_joint_identity']['valid'] and r['weighted_joint_identity']['valid']
    bad = m.gradient_metrics(ge, gf, gj, gj, {'energy': 1., 'force': 3.})
    assert not bad['weighted_joint_identity']['valid']
    z = m.gradient_metrics({'x': np.zeros(2)}, gf, gf, {'x': 3 * gf['x']}, {'energy': 1., 'force': 3.})
    assert z['weighted']['whole']['cosine'] is None
    assert z['weighted']['whole']['cosine_unavailable_reason'] == 'zero_gradient_norm'


@pytest.mark.parametrize('kind', ['gradient', 'finite_difference', 'prediction'])
def test_returned_nonfinite_work_is_known_but_not_valid(tmp_path, kind):
    m = api()
    def finite(x):
        if not np.isfinite(x).all(): raise ValueError('nonfinite')
    journal = m.Journal(tmp_path, lambda: None, ready=lambda x: x, finite=finite)
    with pytest.raises(ValueError): journal.call(kind, 16, lambda: np.array([np.nan]))
    w = json.loads((tmp_path / 'work.json').read_text())
    assert w['counts'][kind]['returned_calls'] == 1
    assert w['counts'][kind]['returned_structure_exposures'] == 16
    assert w['counts'][kind]['valid_calls'] == 0
    assert w['in_flight'] is None


def test_unreturned_call_stays_unknown_and_budget_block_does_not_start_work(tmp_path):
    m = api()
    journal = m.Journal(tmp_path, lambda: None, ready=lambda x: x, finite=lambda x: None)
    def fail(): raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): journal.call('finite_difference', 16, fail)
    w = json.loads((tmp_path / 'work.json').read_text())
    assert w['counts']['finite_difference']['attempted_calls'] == 1
    assert w['counts']['finite_difference']['returned_calls'] == 0
    assert w['in_flight']['structure_count'] == 16
    blocked = tmp_path / 'blocked'; blocked.mkdir()
    def budget(): raise TimeoutError('cap')
    j = m.Journal(blocked, budget, ready=lambda x: x, finite=lambda x: None)
    with pytest.raises(TimeoutError): j.call('gradient', 16, lambda: 1.)
    assert j.work['counts'] == {} and j.work['in_flight'] is None


def test_late_cap_failure_keeps_observations_and_blocks_remaining_cases(tmp_path):
    m = api(); cases = [{'name': 'a'}, {'name': 'b'}]
    def analyzer(case, path): return {'verification_valid': True, 'value': 7.}
    def budget():
        if (tmp_path / 'a/observations.json').exists(): raise TimeoutError('late cap')
    result = m.execute_cases(tmp_path, cases, analyzer, budget)
    assert result['a']['status'] == 'failed' and result['b']['status'] == 'blocked'
    assert json.loads((tmp_path / 'a/observations.json').read_text())['value'] == 7.
    assert not (tmp_path / 'b/observations.json').exists()


def test_report_never_endorses_observation_existence_or_source_integrity_failure(tmp_path):
    m = api(); cases = [{'name': 'a'}, {'name': 'b'}]
    for name in ('a', 'b'): (tmp_path / name).mkdir()
    (tmp_path / 'a/observations.json').write_text('{"verification_valid":true,"value":99}')
    (tmp_path / 'a/summary.json').write_text('{"status":"failed","verification_valid":false}')
    (tmp_path / 'b/summary.json').write_text('{"status":"completed","verification_valid":true}')
    (tmp_path / 'b/observations.json').write_text('{"verification_valid":true,"value":7}')
    assert m.saved_cases(tmp_path, cases, source_integrity=True)['b']['endorsed']
    assert not m.saved_cases(tmp_path, cases, source_integrity=True)['a']['endorsed']
    assert not m.saved_cases(tmp_path, cases, source_integrity=False)['b']['endorsed']


def test_exclusive_run_refuses_repeated_attempt_and_symlink(tmp_path):
    m = api(); path = tmp_path / 'run'
    m.reserve_run(path)
    with pytest.raises(FileExistsError): m.reserve_run(path)
    link = tmp_path / 'link'; link.symlink_to(path, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)): m.reserve_run(link)


def test_fd_retains_both_steps_and_rejects_wrong_derivative(tmp_path):
    import jax.numpy as jnp
    m = api(); params = {'x': jnp.array([1.])}
    ge = {'x': jnp.array([2.])}; gf = {'x': jnp.array([4.])}
    def evaluate(p): return np.array([float(p['x'][0]) ** 2, 2 * float(p['x'][0]) ** 2])
    progress = []
    r = m.finite_differences(params, ge, gf, evaluate, lambda d: progress.append(json.loads(json.dumps(d))), seed=20261002)
    assert r['valid'] and [s['step'] for s in r['energy']['steps']] == [1e-4, 1e-5]
    assert len(progress) == 4
    wrong = m.finite_differences(params, {'x': jnp.array([-2.])}, gf, evaluate, lambda d: None, seed=20261002)
    assert not wrong['valid']


def test_real_weighted_case_matches_historical_predictions_and_preserves_all_state(tmp_path):
    from test_gradient_diagnostics import witness
    from force_consistency.coverage_runs import _partition_predictions, _group_score
    p, model, batch, norm, state = witness(); m = api()
    e, f = _partition_predictions(state['params'], p, model, 16)
    metrics = _group_score(p, e, f); history = {'metrics': {'fitting': metrics, 'anchor': metrics}}
    journal = m.Journal(tmp_path, lambda: None); saved = []
    result = m.analyze_case(state, {'fitting': p, 'anchor': p}, model, norm,
                           {'energy': .25, 'force': 1.75}, history, journal,
                           lambda d: saved.append(json.loads(json.dumps(d))))
    assert result['verification_valid'] and result['state_full_hash_before'] == result['state_full_hash_after']
    assert result['gradients']['weighted_joint_identity']['valid']
    assert result['bias_check']['valid'] and result['finite_differences']['valid']
    assert journal.work['counts']['gradient']['valid_calls'] == 4
    assert journal.work['counts']['finite_difference']['valid_structure_exposures'] == 56
    assert journal.work['counts']['prediction']['valid_structure_exposures'] == 14
    assert (tmp_path / 'predictions.npz').exists() and (tmp_path / 'gradients.msgpack').exists()


def test_actual_saved_state_restore_rejects_wrong_selected_history_hash(tmp_path):
    import shutil
    m = api(); source = Path('runs/aspirin-capacity-2026-10-02/width32')
    if not source.exists(): pytest.skip('Historical local research artifacts unavailable')
    for directory in ('snapshots', 'histories'): (tmp_path / directory).mkdir()
    for name in ('snapshots/0.msgpack', 'snapshots/3700.msgpack', 'histories/3700.json', 'provenance.json'):
        shutil.copyfile(source / name, tmp_path / name)
    prov = json.loads((tmp_path / 'provenance.json').read_text())
    state = m.load_snapshot(tmp_path, 3700, prov)
    assert state['update'] == 3700 and int(state['optimizer']['0']['count']) == 3700
    history = json.loads((tmp_path / 'histories/3700.json').read_text()); history[-1]['state_hashes']['params'] = '0' * 64
    (tmp_path / 'histories/3700.json').write_text(json.dumps(history))
    with pytest.raises(ValueError, match='hash'): m.load_snapshot(tmp_path, 3700, prov)


def test_frozen_config_and_complete_source_pins_guard_before_restore():
    m = api(); config = json.loads(Path('configs/aspirin-capacity-spike.json').read_text())
    changed = {**config, 'microbatch_size': 8}
    with pytest.raises(ValueError, match='configuration'): m.verify_inputs(changed)
    if not Path(config['source_root']).exists(): pytest.skip('Historical local research artifacts unavailable')
    report = m.verify_inputs(config)
    assert report['execution_valid'] and not report['any_learning_accepted']


def test_complete_prelaunch_artifact_set_required_not_supplied_only(tmp_path):
    m = api(); gate = tmp_path / 'gate.json'
    item = tmp_path / 'one'; item.write_text('ok')
    gate.write_text(json.dumps({'status': 'verified', 'artifacts': {str(item): m.digest(item)}}))
    with pytest.raises(ValueError, match='artifact set'):
        m.verify_gate({'prelaunch_gate': str(gate)})


def test_failed_case_report_keeps_rows_but_suppresses_contrasts(tmp_path):
    m = api(); root = tmp_path / 'run'; root.mkdir(); out = tmp_path / 'out'; out.mkdir()
    cases = [{'name': 'a', 'width': 'width32', 'update': 3700}]
    (root / 'a').mkdir()
    (root / 'config.json').write_text(json.dumps({'cases': cases}))
    (root / 'summary.json').write_text(json.dumps({'execution_valid': False, 'source_integrity': False, 'status': 'failed'}))
    (root / 'a/summary.json').write_text('{"status":"failed","verification_valid":false}')
    (root / 'a/observations.json').write_text('{"verification_valid":true,"value":99}')
    result = m.render(root, out)
    assert not result['cases']['a']['endorsed'] and not result['contrasts']
    assert 'failed' in (out / 'capacity-spike-diagnostics.md').read_text().lower()


@pytest.mark.parametrize('failure', ['second_call', 'history_mismatch'])
def test_each_returned_prediction_batch_survives_later_failure(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from force_consistency import coverage_runs as cov, metrics
    m = api(); n = 32 if failure == 'second_call' else 7; calls = []
    ids = tuple(str(i) for i in range(n))
    part = SimpleNamespace(record_ids=ids, positions=np.zeros((n, 1, 3)), forces=np.zeros((n, 1, 3)),
                           energy_residual=np.zeros(n), memberships={i: {'groups': ['a']} for i in ids})
    monkeypatch.setattr(metrics, 'partition_batch', lambda p, ix: {'indices': ix})
    def predict(params, batch):
        calls.append(1)
        if failure == 'second_call' and len(calls) == 2: raise TimeoutError('second prediction interrupted')
        return np.ones(len(batch['indices'])), np.zeros((len(batch['indices']), 3))
    monkeypatch.setattr(cov, '_predict', lambda model: predict)
    monkeypatch.setattr(cov, '_group_score', lambda *args: {'score': 0.})
    journal = m.Journal(tmp_path, lambda: None)
    with pytest.raises((ValueError, TimeoutError)):
        m.analyze_case({'params': {}, 'optimizer': {}, 'update': 3600}, {'fitting': part}, None, None, {},
                       {'metrics': {'fitting': {'score': 1.}}}, journal, lambda d: m.write(tmp_path / 'partial.json', d))
    batches = list((tmp_path / 'prediction-batches').glob('*.npz'))
    assert len(batches) == 1
    with np.load(batches[0], allow_pickle=False) as data:
        assert list(data['record_ids']) == list(ids[:16])
        assert np.all(data['energy_prediction'] == 1.)
    assert json.loads((tmp_path / 'state-audit.json').read_text())['state_unchanged']


def test_abrupt_interruption_has_saved_only_reconstruction_and_unknown_work(tmp_path):
    m = api(); root = tmp_path / 'run'; root.mkdir(); out = tmp_path / 'out'
    cases = [{'name': 'a'}, {'name': 'b'}]
    (root / 'config.json').write_text(json.dumps({'cases': cases}))
    (root / 'a').mkdir()
    pending = {'counts': {'gradient': {'attempted_calls': 1, 'returned_calls': 0}},
               'in_flight': {'kind': 'gradient', 'structure_count': 16, 'completion': 'unknown_until_ready'}}
    (root / 'a/work.json').write_text(json.dumps(pending))
    result = m.render(root, out)
    assert result['summary']['status'] == 'interrupted'
    assert not result['summary']['execution_valid'] and not result['summary']['source_integrity']
    assert result['cases']['a']['summary']['status'] == 'interrupted'
    assert result['cases']['a']['work']['in_flight'] == pending['in_flight']
    assert result['cases']['b']['summary']['status'] == 'blocked'


def test_final_integrity_rechecks_analysis_bytes_after_source_audit(tmp_path, monkeypatch):
    m = api(); config = json.loads(Path('configs/aspirin-capacity-spike.json').read_text())
    if not Path(config['source_root']).exists(): pytest.skip('Historical local artifacts unavailable')
    item = tmp_path / 'analysis.py'; item.write_text('original')
    gate = tmp_path / 'gate.json'; gate.write_text(json.dumps({'status': 'verified', 'artifacts': {str(item): m.digest(item)}}))
    config['prelaunch_gate'] = str(gate)
    from force_consistency.capacity_contract import canonical_hash
    monkeypatch.setattr(m, 'CONFIG_ID', canonical_hash(config)); monkeypatch.setattr(m, 'LAUNCH_INPUTS', {str(item)})
    frozen = json.loads(gate.read_text())
    assert m.verify_execution_integrity(config, expected_gate=frozen)
    item.write_text('drifted')
    with pytest.raises(ValueError): m.verify_execution_integrity(config, expected_gate=frozen)
    gate.write_text(json.dumps({'status': 'verified', 'artifacts': {str(item): m.digest(item)}}))
    with pytest.raises(ValueError, match='gate'):
        m.verify_execution_integrity(config, expected_gate=frozen)
