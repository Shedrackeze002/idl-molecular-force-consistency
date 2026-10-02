"""Frozen capacity protocol and pure guards; no numerical-library imports."""
import hashlib
import json
import math
from pathlib import Path
import platform
import secrets
import struct
from importlib.metadata import version
import msgpack

from .width_rate_contract import canonical_hash, digest, read, safe_path
from .width_rate_replacement_contract import no_symlinks
from .width_rate_replacement import DirectoryLease, exclusive, atomic, utc, _sorted_tree

CONFIG_ID = '888c9307696c1e114dddb3e24384bc30896e4c5d74b4e8279ea623243c40e26e'
PIN_ID = '9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7'
MSE_KEYS = ('energy_mse_eV2', 'force_mse_eV2_per_A2')


def validate_contract(config):
    if canonical_hash(config) != CONFIG_ID or config['source_pins_sha256'] != PIN_ID:
        raise ValueError('Unreviewed capacity configuration')


def validate_case(config, case):
    if case not in config['cases']:
        raise ValueError('Unreviewed capacity case/width assignment')


def verify_pins(pins):
    for name, sha in pins.items():
        path = no_symlinks(name)
        if not path.is_file() or digest(path) != sha:
            raise ValueError('Pinned file changed: ' + name)


def verify_code(code):
    verify_pins(code['code_files'])
    if canonical_hash(code['code_files']) != code['code_fingerprint']:
        raise ValueError('Code fingerprint differs')
    if code['code_files'].get('uv.lock') != code['lock_sha256']:
        raise ValueError('Lock identity differs')
    if len(code['revision']) != 40 or any(c not in '0123456789abcdef' for c in code['revision']):
        raise ValueError('Invalid launch revision')
    if platform.python_version() != code['python'] or any(version(k) != v for k, v in code['dependencies'].items()):
        raise ValueError('Runtime dependency identity differs')


def verify_sources(config):
    validate_contract(config)
    if digest(no_symlinks(config['source_pins'])) != PIN_ID:
        raise ValueError('Source pin manifest differs')
    pins = read(config['source_pins']);verify_pins(pins)
    root = no_symlinks(config['source_directory'])
    p = read(root / 'provenance.json');receipt = read(root / 'commit.json')
    history = read(root / receipt['history']);contract = read(root / 'checkpoint-contract.json')
    summary = read(root / 'summary.json');child = read(root / 'checkpoint-verification.json')
    completion = read(root / 'probe-completion.json');old = read(root / 'config.json')
    expected_model = {**config['model'], 'features': 32}
    if (p['case'] != read(root / 'case.json') or p['case']['features'] != 32 or p['case']['learning_rate'] != .001
        or p['model'] != expected_model or p['config_identity'] != canonical_hash(old)
        or old['loss_weights'] != config['loss_weights'] or old['seed'] != config['seed']
        or old['microbatch_size'] != config['microbatch_size'] or p['precision'] != 'float64' or p['backend'] != 'cpu'
        or not summary['execution_valid'] or summary['status'] != 'completed'
        or receipt != contract['commit'] or receipt['update'] != 6000
        or history[0]['update'] != 0 or history[-1]['update'] != 6000
        or history[0]['metrics'] != p['original_baselines'] or history[-1]['metrics'] != summary['final_metrics']
        or contract['inputs'] != p['inputs'] or contract['normalization'] != p['normalization']
        or digest(root / receipt['checkpoint']) != receipt['checkpoint_sha256']
        or digest(root / receipt['history']) != receipt['history_sha256']
        or child['status'] != 'passed' or child['state_hashes'] != receipt['state_hashes']
        or child['checkpoint_sha256'] != receipt['checkpoint_sha256'] or child['history_sha256'] != receipt['history_sha256']
        or child['adam_count'] != 6000 or child['prediction_frames'] != 144
        or not 0 <= child['maximum_absolute_error'] <= 1e-10
        or completion['status'] != 'passed' or completion['verification_sha256'] != digest(root / 'checkpoint-verification.json')):
        raise ValueError('Authoritative source binding differs')
    fit = p['inputs']['fitting']['record_ids'];anchor = p['inputs']['anchor']['record_ids']
    if len(fit) != 128 or len(set(fit)) != 128 or len(anchor) != 16 or not set(anchor) <= set(fit):
        raise ValueError('Source support changed')
    return {'root': str(root), 'provenance': p, 'pins': pins,
            'input_files_sha256': {r: digest(root / f'{r}-inputs.npz') for r in ('fitting', 'anchor')}}


def _envelope(path, provenance):
    envelope = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    if (set(envelope) != {'payload', 'sha256', 'provenance', 'control_payload', 'control_sha256'}
        or envelope['provenance'] != json.dumps(provenance, sort_keys=True, separators=(',', ':'))
        or hashlib.sha256(envelope['payload']).hexdigest() != envelope['sha256']
        or envelope['control_payload'] != b'\xc0'
        or hashlib.sha256(envelope['control_payload']).hexdigest() != envelope['control_sha256']):
        raise ValueError('Serialized checkpoint envelope differs')
    return msgpack.unpackb(envelope['payload'], raw=False)


def _tree_schema(tree, *, nonnegative=False, zero=False):
    if isinstance(tree, dict):
        return {k: _tree_schema(v, nonnegative=nonnegative, zero=zero) for k, v in tree.items()}
    if not isinstance(tree, msgpack.ExtType) or tree.code != 1:
        raise ValueError('Invalid serialized numerical leaf')
    shape, dtype, raw = msgpack.unpackb(tree.data, raw=False)
    if (dtype != 'float64' or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
        or len(raw) != math.prod(shape) * 8):
        raise ValueError('Numerical shape/dtype differs')
    for (value,) in struct.iter_unpack('=d', raw):
        if not math.isfinite(value):raise ValueError('nonfinite serialized state')
        if nonnegative and value < 0:raise ValueError('Negative Adam variance')
        if zero and value != 0:raise ValueError('Initial Adam moments must be zero')
    return {'shape': shape, 'dtype': dtype}


def _state_metadata(state, update, norm, blueprint=None):
    if set(state) != {'params', 'optimizer', 'update', 'sampler_rng_json', 'model_rng', 'normalization_json'}:
        raise ValueError('Serialized state fields differ')
    if type(state['update']) is not int or state['update'] != update or json.loads(state['normalization_json']) != norm:
        raise ValueError('State update/normalization differs')
    opt = state['optimizer']
    if set(opt) != {'0', '1'} or set(opt['0']) != {'count', 'mu', 'nu'} or opt['1'] != {}:
        raise ValueError('Adam structure differs')
    shape, dtype, raw = msgpack.unpackb(opt['0']['count'].data, raw=False)
    if shape != [] or dtype != 'int32' or len(raw) != 4 or struct.unpack('=i', raw)[0] != update:
        raise ValueError('Adam count differs')
    schema = _tree_schema(state['params'])
    if _tree_schema(opt['0']['mu'], zero=update == 0) != schema or _tree_schema(opt['0']['nu'], nonnegative=True, zero=update == 0) != schema:
        raise ValueError('Adam shape differs')
    if blueprint is not None and schema != blueprint:
        raise ValueError('Width-specific blueprint differs')
    def count(s):return math.prod(s['shape']) if set(s) == {'shape', 'dtype'} else sum(count(v) for v in s.values())
    return {'update': update, 'adam_count': update, 'params_schema': schema, 'parameter_count': count(schema),
            'state_hashes': {k: hashlib.sha256(msgpack.packb(_sorted_tree(state[k]), use_bin_type=True)).hexdigest() for k in ('params', 'optimizer')}}


def audit_state(root, receipt, provenance):
    root = no_symlinks(root);u = receipt['update']
    if receipt['checkpoint'] != f'snapshots/{u}.msgpack' or digest(no_symlinks(root / receipt['checkpoint'])) != receipt['checkpoint_sha256']:
        raise ValueError('State receipt differs')
    zero = _state_metadata(_envelope(root / 'snapshots/0.msgpack', provenance), 0, provenance['normalization'])
    if zero['state_hashes'] != provenance['initial_state_hashes']:
        raise ValueError('Fresh initial state differs')
    actual = _state_metadata(_envelope(root / receipt['checkpoint'], provenance), u, provenance['normalization'], zero['params_schema'])
    if 'state_hashes' in receipt and actual['state_hashes'] != receipt['state_hashes']:
        raise ValueError('Serialized state hashes differ')
    return actual


def probe_binding(root):
    root = no_symlinks(root);receipt = read(root / 'commit.json')
    stat = root.stat()
    names = ['config.json', 'case.json', 'code.json', 'provenance.json', 'snapshots/0.msgpack', 'histories/0.json',
             receipt['checkpoint'], receipt['history'], 'commit.json', 'reference.npz', 'fitting-inputs.npz', 'anchor-inputs.npz']
    return {'root': str(root), 'directory_identity': [stat.st_dev, stat.st_ino],
            'case': read(root / 'case.json'), 'config_identity': canonical_hash(read(root / 'config.json')),
            'files_sha256': {n: digest(no_symlinks(safe_path(root, n))) for n in names}}


def reserve_probe(root, case, config, code):
    validate_case(config, case)
    if read(Path(root) / 'case.json') != case or read(Path(root) / 'config.json') != config or read(Path(root) / 'code.json') != code:
        raise ValueError('Probe reservation case/config/code differs')
    record = {'ordinal': 1, 'token': secrets.token_hex(32), 'binding': probe_binding(root)}
    exclusive(Path(root) / 'probe-attempt.json', record)
    return record


def claim_probe_entry(root, token, *, production=True):
    root = no_symlinks(root);c = read(root / 'config.json');case = read(root / 'case.json')
    if production:
        validate_contract(c);validate_case(c, case)
        if root != (Path.cwd() / c['run_directory'] / case['name']).absolute():
            raise ValueError('Unreviewed probe root')
        verify_sources(c);verify_code(read(root / 'code.json'))
    record = read(root / 'probe-attempt.json')
    if record.get('ordinal') != 1 or record.get('token') != token or record.get('binding') != probe_binding(root):
        raise ValueError('Missing or mismatched probe reservation')
    entry = {'attempt_sha256': digest(root / 'probe-attempt.json'), 'binding': record['binding'], 'entered_at_utc': utc()}
    exclusive(root / 'probe-entry.json', entry)
    return entry
