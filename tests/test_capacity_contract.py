"""Guards against unreviewed protocol, unbound sources and repeated child work."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def contract():
    return importlib.import_module('force_consistency.capacity_contract')


def config():
    return json.loads(Path('configs/aspirin-capacity.json').read_text())


@pytest.mark.parametrize('damage', ['rate', 'width', 'order', 'score', 'pin', 'cap', 'retry'])
def test_unreviewed_numerical_protocol_cannot_enter_production(damage):
    c = config()
    if damage == 'rate': c['learning_rate'] = .001
    if damage == 'width': c['cases'][1]['features'] = 128
    if damage == 'order': c['cases'].reverse()
    if damage == 'score': c['score_updates'].remove(3001)
    if damage == 'pin': c['source_pins_sha256'] = '0' * 64
    if damage == 'cap': c['total_wall_seconds'] += 1
    if damage == 'retry': c['maximum_probe_attempts_per_case'] = 2
    with pytest.raises(ValueError): contract().validate_contract(c)


def test_actual_source_facts_are_verified_without_model_imports():
    m = contract()
    m.validate_contract(config())
    s = m.verify_sources(config())
    assert len(s['provenance']['inputs']['fitting']['record_ids']) == 128
    assert len(s['provenance']['inputs']['anchor']['record_ids']) == 16
    assert s['provenance']['original_baselines']['fitting']['energy_mse_eV2'] == pytest.approx(.03751676427546992)


def test_changed_actual_file_bytes_are_rejected(tmp_path):
    m = contract();p = tmp_path / 'source';p.write_text('fixed')
    pins = {str(p): m.digest(p)};m.verify_pins(pins)
    p.write_text('modified')
    with pytest.raises(ValueError): m.verify_pins(pins)


def test_case_width_assignment_cannot_be_relabelled():
    c = config();case = copy.deepcopy(c['cases'][0]);case['features'] = 64
    with pytest.raises(ValueError): contract().validate_case(c, case)


def fixture_probe_root(tmp_path):
    m = contract();r = tmp_path / 'case';r.mkdir();(r / 'snapshots').mkdir();(r / 'histories').mkdir()
    c = config();case = c['cases'][0]
    for name, value in [('config.json', c), ('case.json', case), ('provenance.json', {'case': case}), ('code.json', {})]:
        (r / name).write_text(json.dumps(value))
    for name in ('snapshots/0.msgpack', 'snapshots/6000.msgpack', 'reference.npz', 'fitting-inputs.npz', 'anchor-inputs.npz'):
        (r / name).write_bytes(name.encode())
    (r / 'histories/0.json').write_text('[]');(r / 'histories/6000.json').write_text('[]')
    receipt = {'update': 6000, 'checkpoint': 'snapshots/6000.msgpack', 'history': 'histories/6000.json',
               'checkpoint_sha256': m.digest(r / 'snapshots/6000.msgpack'), 'history_sha256': m.digest(r / 'histories/6000.json')}
    (r / 'commit.json').write_text(json.dumps(receipt))
    return r, c, case


def test_child_attempt_is_exclusive_and_entry_is_once_only(tmp_path):
    m = contract();r, c, case = fixture_probe_root(tmp_path)
    attempt = m.reserve_probe(r, case, c, {})
    with pytest.raises(FileExistsError): m.reserve_probe(r, case, c, {})
    entry = m.claim_probe_entry(r, attempt['token'], production=False)
    assert entry['attempt_sha256'] == m.digest(r / 'probe-attempt.json')
    with pytest.raises(FileExistsError): m.claim_probe_entry(r, attempt['token'], production=False)


def test_concurrent_child_entries_claim_only_one_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    m = contract();r, c, case = fixture_probe_root(tmp_path);a = m.reserve_probe(r, case, c, {})
    def enter():
        try:m.claim_probe_entry(r, a['token'], production=False);return 'entered'
        except FileExistsError:return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:results = list(pool.map(lambda _:enter(), range(2)))
    assert sorted(results) == ['entered','rejected']


def test_replaced_root_cannot_reuse_prior_attempt_binding(tmp_path):
    m = contract();r, c, case = fixture_probe_root(tmp_path);a = m.reserve_probe(r, case, c, {})
    retained = tmp_path / 'retained';r.rename(retained);shutil.copytree(retained,r)
    with pytest.raises(ValueError):m.claim_probe_entry(r, a['token'], production=False)
    assert not (r / 'probe-entry.json').exists()


@pytest.mark.parametrize('damage', ['token', 'zero', 'terminal', 'reference', 'case', 'code'])
def test_forged_or_changed_child_binding_is_rejected_before_entry(tmp_path, damage):
    m = contract();r, c, case = fixture_probe_root(tmp_path)
    a = m.reserve_probe(r, case, c, {})
    if damage == 'token': token = 'wrong'
    else:
        token = a['token']
        name = {'zero': 'snapshots/0.msgpack', 'terminal': 'snapshots/6000.msgpack',
                'reference': 'reference.npz', 'case': 'case.json', 'code': 'code.json'}[damage]
        (r / name).write_text('changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        m.claim_probe_entry(r, token, production=False)
    assert not (r / 'probe-entry.json').exists()


def test_public_missing_probe_reservation_never_imports_jax(tmp_path):
    m = contract();r, c, case = fixture_probe_root(tmp_path)
    code = "import sys;from force_consistency.capacity_contract import claim_probe_entry\ntry:claim_probe_entry(sys.argv[1],'missing')\nexcept Exception:pass\nassert 'jax' not in sys.modules\nassert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, '-c', code, str(r)], check=True, capture_output=True)
    assert not (r / 'probe-entry.json').exists()


def test_symlinked_child_artifacts_are_rejected(tmp_path):
    m = contract();r, c, case = fixture_probe_root(tmp_path)
    external = tmp_path / 'external';external.write_bytes(b'foreign')
    (r / 'reference.npz').unlink();(r / 'reference.npz').symlink_to(external)
    with pytest.raises(ValueError): m.reserve_probe(r, case, c, {})


def test_actual_serialized_initial_and_terminal_states_match_width_blueprint():
    m = contract();root = Path(config()['source_directory']);prov = m.read(root / 'provenance.json')
    receipt = m.read(root / 'commit.json')
    got = m.audit_state(root, receipt, prov)
    assert got['update'] == got['adam_count'] == 6000
    assert got['state_hashes'] == receipt['state_hashes']
    assert got['parameter_count'] == 13569


@pytest.mark.parametrize('damage,match', [('nan_params','nonfinite'), ('negative_variance','variance'), ('count','count'), ('nonzero_initial_moments','zero')])
def test_locally_rehashed_invalid_state_is_still_rejected(tmp_path, damage, match):
    import hashlib
    import msgpack
    import struct
    m = contract();source = Path(config()['source_directory']);root = tmp_path / 'state';(root / 'snapshots').mkdir(parents=True)
    for u in (0,6000):shutil.copy2(source / f'snapshots/{u}.msgpack', root / f'snapshots/{u}.msgpack')
    prov = m.read(source / 'provenance.json');receipt = m.read(source / 'commit.json')
    u = 0 if damage == 'nonzero_initial_moments' else 6000
    path = root / f'snapshots/{u}.msgpack';envelope = msgpack.unpackb(path.read_bytes(),raw=False)
    state = msgpack.unpackb(envelope['payload'],raw=False)
    if damage == 'count':state['optimizer']['0']['count'] = msgpack.ExtType(1,msgpack.packb([[],'int32',struct.pack('=i',6001)],use_bin_type=True))
    else:
        tree = state['params'] if damage == 'nan_params' else state['optimizer']['0']['nu' if damage == 'negative_variance' else 'mu']
        while isinstance(next(iter(tree.values())),dict):tree = next(iter(tree.values()))
        k = next(iter(tree));shape,dtype,raw = msgpack.unpackb(tree[k].data,raw=False)
        value = float('nan') if damage == 'nan_params' else -1. if damage == 'negative_variance' else .5
        tree[k] = msgpack.ExtType(1,msgpack.packb([shape,dtype,struct.pack('=d',value)+raw[8:]],use_bin_type=True))
    envelope['payload'] = msgpack.packb(state,use_bin_type=True);envelope['sha256'] = hashlib.sha256(envelope['payload']).hexdigest()
    path.write_bytes(msgpack.packb(envelope,use_bin_type=True))
    receipt = {**receipt,'update':u,'checkpoint':f'snapshots/{u}.msgpack','checkpoint_sha256':m.digest(path)}
    with pytest.raises(ValueError,match=match):m.audit_state(root,receipt,prov)
