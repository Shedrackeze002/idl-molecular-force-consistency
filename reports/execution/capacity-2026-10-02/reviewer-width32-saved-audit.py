"""Independent saved-only first-arm audit. No model calls or global-run reads."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from force_consistency import capacity_contract as cc
from force_consistency import capacity_report as cr

start = time.perf_counter()
root = Path('runs/aspirin-capacity-2026-10-02/width32')
outpath = Path('reports/execution/capacity-2026-10-02/reviewer-width32-saved-verification.json')
def hashes():
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}
before = hashes()
c = cc.read('configs/aspirin-capacity.json')
cc.validate_contract(c)
source = cc.verify_sources(c)
validated = cr._validated_case(root, c, c['cases'][0], source=source)
p = cc.read(root / 'provenance.json')
h = cc.read(root / 'histories/6000.json')
independent = {}
with np.load(root / 'reference.npz', allow_pickle=False) as ref:
    for role in ('fitting', 'anchor'):
        with np.load(root / (role + '-inputs.npz'), allow_pickle=False) as data:
            er = ref[role + '_energy'] - data['energy_residual']
            fr = ref[role + '_forces'] - data['forces']
        assert np.isfinite(er).all() and np.isfinite(fr).all()
        metrics = {'energy_mse_eV2': float(np.mean(er ** 2)),
                   'force_mse_eV2_per_A2': float(np.mean(fr ** 2))}
        groups = sorted({g for m in p['inputs'][role]['memberships'].values() for g in m['groups']})
        per_group = {}
        for g in groups:
            idx = [i for i, rid in enumerate(p['inputs'][role]['record_ids'])
                   if g in p['inputs'][role]['memberships'][rid]['groups']]
            per_group[g] = {'energy_mse_eV2': float(np.mean(er[idx] ** 2)),
                            'force_mse_eV2_per_A2': float(np.mean(fr[idx] ** 2))}
            for key, value in per_group[g].items():
                assert np.isclose(value, h[-1]['metrics'][role]['per_group'][g][key], atol=1e-12, rtol=1e-10)
        equal_group = {k: float(np.mean([v[k] for v in per_group.values()])) for k in metrics}
        for k in metrics:
            assert np.isclose(metrics[k], h[-1]['metrics'][role][k], atol=1e-12, rtol=1e-10)
            assert np.isclose(equal_group[k], h[-1]['metrics'][role]['equal_group'][k], atol=1e-12, rtol=1e-10)
        fractions = {k: metrics[k] / p['original_baselines'][role][k] for k in metrics}
        accepted = all(v < c['gate_fraction'] for v in fractions.values())
        assert accepted == h[-1][role + '_gate']['accepted']
        decomposition = {'mean_error_eV': float(er.mean()), 'mean_squared_eV2': float(er.mean() ** 2),
                         'centered_variance_eV2': float(er.var())}
        assert np.isclose(decomposition['mean_squared_eV2'] + decomposition['centered_variance_eV2'], metrics['energy_mse_eV2'], atol=1e-12, rtol=1e-10)
        spikes = {}
        for k in metrics:
            spikes[k] = []
            for i, row in enumerate(h[1:], 1):
                fraction = row['metrics'][role][k] / p['original_baselines'][role][k]
                if fraction > 1:
                    spikes[k].append({'update': row['update'], 'raw_mse': row['metrics'][role][k],
                                      'baseline_fraction': fraction,
                                      'next_saved_update': h[i+1]['update'] if i+1 < len(h) else None,
                                      'next_saved_fraction': h[i+1]['metrics'][role][k] / p['original_baselines'][role][k] if i+1 < len(h) else None})
        independent[role] = {'metrics': metrics, 'fractions': fractions, 'accepted': accepted,
                             'per_group': per_group, 'equal_group': equal_group,
                             'decomposition': decomposition, 'post_initial_above_baseline_samples': spikes}
gate = cc.read('reports/execution/capacity-2026-10-02/preproduction-gate.json')
code = cc.read(root / 'code.json')
assert code['code_files'] == gate['suite']['code']['code_files']
assert code['code_fingerprint'] == gate['code_fingerprint']
assert code['dependencies'] == gate['suite']['code']['dependencies']
for name, sha in gate['verification_artifacts'].items():
    assert cc.digest(name) == sha
assert code['revision'] == 'b45ff2a66923af14a51dce7190b3808810b2a478'
after = hashes()
assert before == after
out = {'status': 'passed_first_arm_only', 'scope': 'Completed width32 only; width64/global closure not inspected.',
       'no_model_work': True, 'elapsed_seconds': time.perf_counter() - start,
       'case_file_count': len(before), 'case_file_hashes': before,
       'code': {'revision': code['revision'], 'fingerprint': code['code_fingerprint']},
       'suite_evidence': {'passed': gate['suite']['passed'], 'seconds': gate['suite']['elapsed_seconds'],
                          'reused_saved_evidence': True},
       'validated_case': {k: v for k, v in validated.items() if k != 'history'},
       'independent_endpoint': independent,
       'trials': [{'status': 'inspection_output_truncated', 'action': 'Combined plan/review/evidence read', 'resolution': 'Focused reads retained key controls and records.'},
                  {'status': 'inspection_output_truncated', 'action': 'Printed full support memberships', 'resolution': 'No numerical conclusion drawn from truncated output; verification uses complete saved files.'},
                  {'status': 'passed', 'action': 'Saved first-arm audit plus independent NumPy endpoint/group/gate/spike recomputation'}]}
outpath.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'status': out['status'], 'case_file_count': len(before),
                  'endpoint': {r: {k: v for k,v in m.items() if k in ('metrics','fractions','accepted')} for r,m in independent.items()},
                  'sampled_peaks': validated['sampled_peaks'], 'saved_verification': str(outpath)}, sort_keys=True))
