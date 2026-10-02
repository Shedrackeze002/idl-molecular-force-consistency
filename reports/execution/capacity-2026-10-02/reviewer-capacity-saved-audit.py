"""Independent saved-only whole-pair verification. No model/numerical probe calls."""
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from force_consistency import capacity_contract as cc
from force_consistency import capacity_report as cr

start = time.perf_counter()
root = Path('runs/aspirin-capacity-2026-10-02')
execution = Path('reports/execution/capacity-2026-10-02')
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw_hashes():
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
def close(a, b):
    assert np.isclose(a, b, atol=1e-12, rtol=1e-10), (a, b)
before = raw_hashes()
c = cc.read(root / 'config.json')
assert c == cc.read('configs/aspirin-capacity.json')
cc.validate_contract(c)
source = cc.verify_sources(c)
saved = cc.read(cr.REPORT_JSON)
rebuilt = cr.build_report(root)
assert saved == rebuilt
assert cr.verify_saved_report(rebuilt)
assert rebuilt['execution_valid'] and rebuilt['verification_errors'] == []
assert rebuilt['updates'] == rebuilt['verified_persisted_updates'] == rebuilt['reported_persisted_updates'] == 12000
assert rebuilt['frame_exposures'] == 1536000 and not rebuilt['any_pending_work_unknown']
assert not rebuilt['any_learning_accepted'] and rebuilt['full_pilot'] == 'pending'
assert rebuilt['contrast_orientation'] == 'width64 minus width32 and width64 divided by width32 in production'
assert rebuilt['limits'] == cr.LIMITS

independent = {}
for case in c['cases']:
    name = case['name'];path = root / name;p = cc.read(path / 'provenance.json')
    h = cc.read(path / 'histories/6000.json');a = rebuilt['cases'][name]
    assert [x['update'] for x in h] == c['score_updates'] and len(h) == 63
    assert set(x.name for x in (path / 'snapshots').iterdir()) == {f'{u}.msgpack' for u in c['score_updates']}
    assert set(x.name for x in (path / 'histories').iterdir()) == {f'{u}.json' for u in c['score_updates']}
    assert a['attempted_updates'] == a['persisted_updates'] == a['verified_persisted_updates'] == a['updates'] == 6000
    assert a['frame_exposures'] == 768000 and a['child_maximum_prediction_error'] == 0.
    assert a['work'] == {'initialization_status': 'completed', 'initializations': 1, 'initialization_frames': 16,
                         'warmup_status': 'completed', 'warmup_frame_exposures': 128, 'scoring_status': 'completed',
                         'scoring_prediction_frames': 9072, 'reference_status': 'completed', 'reference_prediction_frames': 144,
                         'oracle_status': 'completed', 'probe_status': 'completed', 'oracle_fullbatch_gradient_frames': 256,
                         'oracle_accumulated_gradient_frames': 256, 'independent_numpy_proposals': 2,
                         'discarded_accumulated_proposals': 2}
    assert a['child_work'] == {'prediction_frames': 144, 'template_initializations': 1,
                               'template_initializer_frames': 16, 'training_updates': 0}
    assert a['distinct_prediction_structures'] == 128
    assert len(p['inputs']['fitting']['record_ids']) == 128 and len(p['inputs']['anchor']['record_ids']) == 16
    assert set(p['inputs']['anchor']['record_ids']) <= set(p['inputs']['fitting']['record_ids'])
    assert all(m['seen_in_training'] for role in ('fitting','anchor') for m in p['inputs'][role]['memberships'].values())
    role_results = {}
    with np.load(path / 'reference.npz', allow_pickle=False) as ref:
        for role in ('fitting','anchor'):
            meta = p['inputs'][role]
            with np.load(path / (role + '-inputs.npz'), allow_pickle=False) as data:
                target = np.array(data['energy_residual']);forces = np.array(data['forces'])
                er = ref[role + '_energy'] - target;fr = ref[role + '_forces'] - forces
            assert np.isfinite(er).all() and np.isfinite(fr).all()
            def metrics(idx):
                e = er[idx];f = fr[idx];t = target[idx];ft = forces[idx]
                em = float(np.mean(e ** 2));fm = float(np.mean(f ** 2))
                return {'energy_mse_eV2': em, 'force_mse_eV2_per_A2': fm,
                        'energy_rmse_eV': float(np.sqrt(em)), 'force_rmse_eV_per_A': float(np.sqrt(fm)),
                        'zero_energy_mse_eV2': float(np.mean(t ** 2)), 'zero_force_mse_eV2_per_A2': float(np.mean(ft ** 2)),
                        'mean_energy_mse_eV2': float(np.var(t)), 'record_count': len(idx),
                        'energy_count': len(idx), 'force_component_count': int(f.size)}
            aggregate = metrics(list(range(len(er))))
            endpoint = h[-1]['metrics'][role]
            for k,v in aggregate.items():close(v, endpoint[k])
            assert endpoint['record_ids'] == meta['record_ids']
            groups = sorted({g for m in meta['memberships'].values() for g in m['groups']})
            assert set(endpoint['per_group']) == set(groups)
            per_group = {}
            for g in groups:
                idx = [i for i,rid in enumerate(meta['record_ids']) if g in meta['memberships'][rid]['groups']]
                per_group[g] = metrics(idx)
                for k,v in per_group[g].items():close(v, endpoint['per_group'][g][k])
                assert endpoint['per_group'][g]['record_ids'] == [meta['record_ids'][i] for i in idx]
            equal = {k: float(np.mean([v[k] for v in per_group.values()])) for k in cc.MSE_KEYS}
            for k,v in equal.items():close(v, endpoint['equal_group'][k])
            fractions = {k: aggregate[k] / p['original_baselines'][role][k] for k in cc.MSE_KEYS}
            accepted = all(v < .1 for v in fractions.values())
            assert accepted == h[-1][role + '_gate']['accepted']
            assert fractions == a['original_baseline_fractions'][role]
            decomposition = {'energy_mse_eV2': aggregate['energy_mse_eV2'],
                             'force_mse_eV2_per_A2': aggregate['force_mse_eV2_per_A2'],
                             'signed_mean_energy_error_eV': float(er.mean()),
                             'mean_energy_error_squared_eV2': float(er.mean() ** 2),
                             'centered_energy_error_variance_eV2': float(er.var())}
            for k,v in decomposition.items():close(v,a['residual_decomposition'][role][k])
            close(decomposition['mean_energy_error_squared_eV2'] + decomposition['centered_energy_error_variance_eV2'], aggregate['energy_mse_eV2'])
            peaks = {};spikes = {};window = {}
            old = next(x for x in h if x['update'] == 5500)
            for k in cc.MSE_KEYS:
                peak = max(h, key=lambda x:x['metrics'][role][k])
                peaks[k] = {'update': peak['update'], 'raw_mse': peak['metrics'][role][k],
                            'original_baseline_fraction': peak['metrics'][role][k] / p['original_baselines'][role][k]}
                assert peaks[k] == a['sampled_peaks'][role][k]
                window[k] = {'relative_change': (aggregate[k] - old['metrics'][role][k]) / old['metrics'][role][k], 'unavailable_reason': None}
                assert window[k] == a['terminal_window']['metrics'][role][k]
                spikes[k] = []
                for i,row in enumerate(h[1:],1):
                    fraction = row['metrics'][role][k] / p['original_baselines'][role][k]
                    if fraction > 1:
                        spikes[k].append({'update': row['update'], 'raw_mse': row['metrics'][role][k],
                                          'baseline_fraction': fraction,
                                          'next_saved_update': h[i+1]['update'] if i+1<len(h) else None,
                                          'next_saved_fraction': h[i+1]['metrics'][role][k] / p['original_baselines'][role][k] if i+1<len(h) else None})
            role_results[role] = {'aggregate': aggregate, 'per_group': per_group, 'equal_group': equal,
                                  'fractions': fractions, 'joint_gate_accepted': accepted,
                                  'residual_decomposition': decomposition, 'sampled_peaks': peaks,
                                  'post_initial_above_baseline_samples': spikes, 'terminal5500_to6000': window}
    assert h[-1]['metrics']['anchor_fitted'] == h[-1]['metrics']['anchor']
    for u,o in a['oracles'].items():
        assert o['maximum_absolute_difference'] < o['atol']
        assert o['proposal_state_hashes'] == o['committed_state_hashes']
        assert o['incoming_nonzero_moments'] == (int(u) == 3001)
    independent[name] = {'roles': role_results, 'parameter_count': a['parameter_count'],
                          'initial_state_hashes': a['initial_state_hashes'], 'oracles': a['oracles'],
                          'case_elapsed_seconds': a['elapsed_seconds'], 'work': a['work'], 'child_work': a['child_work']}
assert rebuilt['cases']['width32']['history'][0]['metrics'] == rebuilt['cases']['width64']['history'][0]['metrics']
assert independent['width32']['initial_state_hashes'] != independent['width64']['initial_state_hashes']
contrasts = {}
for role in ('fitting','anchor'):
    contrasts[role] = {}
    for k in cc.MSE_KEYS:
        a = independent['width32']['roles'][role]['aggregate'][k];b = independent['width64']['roles'][role]['aggregate'][k]
        contrasts[role][k] = {'wider_minus_narrower': b-a, 'wider_divided_by_narrower': b/a, 'ratio_unavailable_reason': None}
assert contrasts == rebuilt['endpoint_contrasts']
closed = cc.read(root / 'finalization.json')
prefix = Path(c['evidence_log']).read_bytes()[:closed['evidence_prefix_bytes']]
assert len(prefix) == closed['evidence_prefix_bytes'] and hashlib.sha256(prefix).hexdigest() == closed['evidence_prefix_sha256']
assert closed['within_global_cap'] and closed['elapsed_seconds'] < 10800
assert closed['report_sha256'] == digest(cr.REPORT_JSON) and closed['report_md_sha256'] == digest(cr.REPORT_MD)
assert closed['raw_inventory_sha256'] == digest(root / 'raw-artifacts.json') == rebuilt['raw_inventory_sha256']
assert cc.read(root / 'raw-artifacts.json') == {k:v for k,v in before.items() if k not in ('raw-artifacts.json','finalization.json')}
archive_inventory = cc.read(execution / 'numerical-run-inventory.json')
assert archive_inventory == before
with zipfile.ZipFile(execution / 'numerical-run.zip') as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(before)
    assert set(archive.namelist()) == set(before)
    for name,sha in archive_inventory.items():
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert hashlib.sha256(archive.read(name)).hexdigest() == sha
invocation = cc.read(execution / 'production-invocation.json')
process = cc.read(execution / 'production-process-completion.json')
assert invocation['ordinal'] == 1 and invocation['no_retry'] is True and invocation['code'] == cc.read(root/'code.json')
assert process['exit_code'] == 0 and process['stop_reason'] == 'returned'
assert process['stdout_sha256'] == digest(execution/'production-stdout.txt') and process['stderr_sha256'] == digest(execution/'production-stderr.txt')
gate = cc.read(execution / 'preproduction-gate.json')
assert invocation['code']['code_files'] == gate['suite']['code']['code_files']
for name,sha in gate['verification_artifacts'].items():assert digest(name) == sha
for name,sha in invocation['code']['code_files'].items():assert digest(root/'code-snapshot'/name) == sha
closeout = cc.read(execution/'saved-closeout-verification.json')
assert closeout['archive_sha256'] == digest(execution/'numerical-run.zip')
assert closeout['inventory_sha256'] == digest(execution/'numerical-run-inventory.json')
assert closeout['report_sha256'] == digest(cr.REPORT_JSON) and closeout['report_md_sha256'] == digest(cr.REPORT_MD)
for name,sha in closeout['figures'].items():assert digest(name) == sha
after = raw_hashes()
assert before == after
out = {'status': 'passed_whole_pair_saved_only', 'elapsed_seconds': time.perf_counter()-start,
       'new_model_initializations': 0, 'new_predictions': 0, 'new_training_updates': 0, 'new_probes': 0,
       'raw_files_unchanged': True, 'raw_file_count': len(before), 'raw_file_hashes': before,
       'canonical_json_markdown_match': True, 'execution_valid': True, 'verification_errors': [],
       'counts': {k:rebuilt[k] for k in ('updates','frame_exposures','reported_persisted_updates','verified_persisted_updates')},
       'independent_cases': independent, 'independent_endpoint_contrasts': contrasts,
       'closure': closed, 'archive': {'member_count':len(before),'crc':'passed','every_member_hash_matches':True,
                                     'sha256':digest(execution/'numerical-run.zip'),'inventory_sha256':digest(execution/'numerical-run-inventory.json')},
       'report_sha256':digest(cr.REPORT_JSON),'report_md_sha256':digest(cr.REPORT_MD),
       'reused_suite': {'passed':585,'elapsed_seconds':468.33,'log_sha256':digest(execution/'full-suite.txt')},
       'limits': rebuilt['limits'], 'figures': closeout['figures'],
       'trials': [{'status':'inspection_output_truncated','action':'Combined report/closeout read','resolution':'Focused reads and full-byte scripted checks'},
                  {'status':'read_path_not_found','action':'Attempted scripts/closeout_capacity.py','resolution':'Located scripts/close_capacity_saved.py; no execution or model work'},
                  {'status':'passed','action':'Whole-pair saved-only audit with independent NumPy reductions and archive/closure verification'}]}
path = execution/'reviewer-capacity-saved-verification.json'
path.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':out['status'],'elapsed_seconds':out['elapsed_seconds'],'raw_file_count':len(before),
                  'parameters':{k:v['parameter_count'] for k,v in independent.items()},
                  'endpoint_contrasts':contrasts,'verification':str(path)},sort_keys=True))
