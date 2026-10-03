"""Independent read-only plan evidence checks; no model or numerical runtime imports."""
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'src'))
from force_consistency import capacity_contract as cc
from report_offset_profile import audit_saved, raw_full_hash, score_arrays
import numpy as np

start = time.perf_counter()
read = cc.read
planning = REPO / 'reports/execution/force-isolation-2026-10-03'
pins = read(planning / 'source-pins.json')
context = read(planning / 'context-paths.json')
old_map = read(REPO / 'reports/execution/offset-profile-2026-10-02/source-pins.json')
inventory = read(REPO / 'reports/execution/offset-profile-2026-10-02/numerical-run-inventory.json')
gate = read(REPO / 'reports/execution/offset-profile-2026-10-02/prelaunch-gate.json')
control = REPO / 'runs/aspirin-offset-profile-2026-10-02'
expected = set(old_map) | {'runs/aspirin-offset-profile-2026-10-02/' + n for n in inventory} | set(gate['artifacts']) | set(context)
assert len(context) == 15 and len(old_map) == 601 and len(inventory) == 18324
assert len(pins) == 18963 and set(pins) == expected
assert all(pins[n] == value for n, value in old_map.items())
assert all(pins['runs/aspirin-offset-profile-2026-10-02/' + n] == value for n, value in inventory.items())
assert all(pins[n] == value for n, value in gate['artifacts'].items())
cc.verify_pins(pins)
current = {str(p.relative_to(control)) for p in control.rglob('*') if p.is_file()}
extras = current - set(inventory)
assert extras == {'.DS_Store'} and not set(inventory) - current
for p in control.rglob('*'):
    assert not p.is_symlink()
settings = read(planning / 'proposed-settings.json')
facts = read(planning / 'source-context-verification.json')
assert cc.digest(control / '.DS_Store') == facts['current_control_extras']['.DS_Store']
with (REPO / 'reports/PAPER_EVIDENCE_LOG.md').open('rb') as stream:
    prefix_sha = hashlib.sha256(stream.read(facts['protected_evidence_prefix_bytes'])).hexdigest()
assert prefix_sha == facts['protected_evidence_prefix_sha256']
audit = audit_saved(control)
p = read(control / 'provenance.json')
source = REPO / settings['source_root']
original = read(source / 'provenance.json')
zero = cc._envelope(source / 'snapshots/0.msgpack', original)
projected = cc._envelope(control / 'snapshots/0.msgpack', p)
assert raw_full_hash(zero) == facts['source_zero_full_hash']
assert raw_full_hash(projected) == facts['control_projected_zero_full_hash'] == settings['initial_projected_full_hash_must_equal']
assert p['inputs'] == original['inputs']
assert p['normalization'] == original['normalization']
assert p['model'] == original['model']
grid = [r['update'] for r in audit['history']]
assert grid == settings['score_updates'] and len(grid) == 63
postzero = [u for u in grid if u > 0]
assert len(postzero) == 62 and 3001 in postzero
initial_forces_equal = {}
with np.load(control / 'replay/shared-zero.npz', allow_pickle=False) as unprojected, np.load(control / 'predictions/0.npz', allow_pickle=False) as calibrated:
    for role in ('fitting', 'anchor'):
        assert np.array_equal(unprojected[role + '_forces'], calibrated[role + '_forces'])
        initial_forces_equal[role] = True
assert not any(n == 'jax' or n.startswith('jax.') for n in sys.modules)
artifacts = [
    'docs/superpowers/specs/2026-10-03-force-isolation-design.md',
    'docs/superpowers/plans/2026-10-03-force-isolation.md',
    'reports/execution/force-isolation-2026-10-03/proposed-settings.json',
    'reports/execution/force-isolation-2026-10-03/context-paths.json',
    'reports/execution/force-isolation-2026-10-03/source-context-verification.json',
    'reports/execution/force-isolation-2026-10-03/source-pins.json',
    'reports/execution/force-isolation-2026-10-03/review-request.json',
    'reports/execution/force-isolation-2026-10-03/planning-record.json',
    'scripts/offset_profile_core.py', 'scripts/train_offset_profile.py',
    'scripts/run_offset_profile.py', 'scripts/report_offset_profile.py',
    'src/force_consistency/capacity_contract.py', 'src/force_consistency/loss_balance.py',
    'src/force_consistency/width_rate_runs.py', 'src/force_consistency/runs.py',
]
out = {
    'status': 'PASS_saved_only_plan_context', 'source_pin_count': len(pins),
    'source_membership_exact_union': True, 'source_hashes_verified': True,
    'context_path_count': len(context), 'control_numerical_members': len(inventory),
    'control_declared_extras': {n: cc.digest(control / n) for n in extras},
    'protected_prefix_bytes': facts['protected_evidence_prefix_bytes'],
    'protected_prefix_sha256': prefix_sha, 'scored_states': len(grid),
    'postzero_grid': postzero, 'saved_control_audit': 'PASS',
    'accepted_projection_receipts': audit['accepted_projection_receipts'],
    'control_gates': audit['recomputed_gates'],
    'source_zero_full_hash': raw_full_hash(zero),
    'projected_zero_full_hash': raw_full_hash(projected),
    'projected_initial_force_arrays_bitwise_equal': initial_forces_equal,
    'model_calls': 0, 'training_updates': 0, 'initializations': 0,
    'test_suite_runs': 0, 'jax_imported': False,
    'artifact_sha256': {n: cc.digest(REPO / n) for n in artifacts},
    'elapsed_saved_only_seconds': time.perf_counter() - start,
}
(Path(__file__).parent / 'saved-context-check.json').write_text(json.dumps(out, sort_keys=True, indent=2) + '\n')
print(json.dumps({k: out[k] for k in ('status', 'source_pin_count', 'scored_states', 'accepted_projection_receipts', 'jax_imported', 'elapsed_saved_only_seconds')}, indent=2))
