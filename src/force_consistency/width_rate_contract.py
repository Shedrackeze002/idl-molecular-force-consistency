"""Reviewed immutable width/rate controls. Standard library only; safe for reporting."""
import hashlib
import json
import math
from pathlib import Path

CONFIG_ID='e64165f0e511ae073dae1fb6e3da49053fee110e5844b5da0471bce47ec01198'
PINS_ID='621e60dd590f18214ac6467f9b9f00eff7d1248067eb1d2c9ee19f014362e0c4'
MSE_KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')
OLD_ROOT=Path('runs/aspirin-loss-balance-2026-09-30')
CONT_ROOT=Path('runs/aspirin-convergence-budget-2026-09-30')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def expected_updates():return sorted({0,1,250,2001,*range(100,6001,100)})


def validate_contract(config):
    if canonical_hash(config)!=CONFIG_ID:raise ValueError('Unreviewed width/rate configuration')
    if config['source_pin_manifest_sha256']!=PINS_ID:raise ValueError('Unreviewed source pin identity')


def safe_path(root,name):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe artifact path')
    root=Path(root).resolve();path=(root/rel).resolve()
    if not path.is_relative_to(root):raise ValueError('Artifact leaves its root')
    return path


def join_histories(original,continuation):
    if [r['update'] for r in original]!=sorted({0,250,*range(100,2001,100)}):raise ValueError('Original history incomplete')
    if [r['update'] for r in continuation]!=sorted({2000,2001,*range(2100,6001,100)}):raise ValueError('Continuation history incomplete')
    if original[-1]['metrics']!=continuation[0]['metrics']:raise ValueError('Historical duplicate2000 mismatch')
    joined={r['update']:r['metrics'] for r in [*original,*continuation[1:]]}
    for metrics in joined.values():
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:
                value=metrics[role][key]
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Invalid historical raw MSE')
    return joined


def verify_width_rate_sources(config):
    validate_contract(config);path=Path(config['source_pin_manifest'])
    if digest(path)!=PINS_ID:raise ValueError('Source pin map substituted')
    pins=read(path)
    if len(pins['files_sha256'])!=326:raise ValueError('Incomplete immutable source allowlist')
    for name,want in pins['files_sha256'].items():
        if digest(safe_path(Path.cwd(),name))!=want:raise ValueError(f'Source changed: {name}')
    old=read(OLD_ROOT/'balanced/history.json');continued=read(CONT_ROOT/'balanced/history.json')
    historical=join_histories(old,continued)
    original_baselines=old[0]['metrics']
    oldsummary=read(OLD_ROOT/'balanced/summary.json');newsummary=read(CONT_ROOT/'balanced/summary.json')
    if oldsummary['initial_metrics']!=original_baselines or newsummary['original_baselines']!=original_baselines:
        raise ValueError('Historical original baseline mismatch')
    if not oldsummary['execution_valid'] or not newsummary['execution_valid']:raise ValueError('Historical execution invalid')
    return {'config_identity':CONFIG_ID,'pin_manifest_identity':PINS_ID,'pin_count':326,
            'pins':pins['files_sha256'],'historical':historical,'original_baselines':original_baselines,
            'historical_config':read(config['historical_config']),
            'historical_initial_state_hashes':oldsummary['initial_state_hashes'],
            'historical_inputs':read(OLD_ROOT/'balanced/provenance.json')['inputs'],
            'normalization':read(OLD_ROOT/'balanced/config.json')['normalization']}


def resolve_commit(root):
    """Only a published receipt defines persistence; uncommitted files are irrelevant."""
    receipt=read(Path(root)/'commit.json')
    if set(receipt)!= {'update','checkpoint','history','checkpoint_sha256','history_sha256','state_hashes'}:raise ValueError('Invalid commit fields')
    for key in ('checkpoint','history'):
        if digest(safe_path(root,receipt[key]))!=receipt[key+'_sha256']:raise ValueError('Committed artifact hash mismatch')
    history=read(safe_path(root,receipt['history']))
    if not history or history[-1]['update']!=receipt['update']:raise ValueError('Committed history endpoint mismatch')
    return receipt,history
