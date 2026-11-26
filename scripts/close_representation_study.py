"""Audit retained representation outputs without importing model frameworks."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import msgpack
import numpy as np

ATOL, RTOL = 1e-14, 1e-12
GROUPS = {'13', '18', '18_2', '66', '78'}
CANDIDATES = {f'{family}-{seed}' for family in ('E64', 'S64', 'SP') for seed in (0, 1)}
COUNTS = {'E64': 43521, 'SP': 43473, 'S64': 25473}
SPLIT = '73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(data, parse_constant=reject)


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def assert_close(actual, expected, field='metric'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), field + ' keys')
        for key in expected:
            assert_close(actual[key], expected[key], field + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), field + ' support')
        for index, value in enumerate(expected):
            assert_close(actual[index], value, field + '.' + str(index))
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), field + ' reduction differs')
    else:
        require(type(actual) is type(expected) and actual == expected, field + ' differs')


class BoundInputs:
    def __init__(self, repository, retained_root, inventory):
        self.roots = {'repository': Path(repository).resolve(), 'retained': Path(retained_root).resolve()}
        self.files = inventory['files']

    def path(self, key):
        require(key in self.files, 'Unlisted input: ' + key)
        domain, name = key.split(':', 1)
        relative = Path(name)
        require(domain in self.roots and not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe input identity')
        require(relative.name not in ('train.npz', 'test.npz', 'Landscape17.zip'), 'Disallowed numerical input')
        path = self.roots[domain] / relative
        require(not path.is_symlink() and path.is_file()
                and path.resolve().is_relative_to(self.roots[domain]), 'Input path unavailable: ' + key)
        return path

    def read(self, key):
        data = self.path(key).read_bytes()
        expected = self.files[key]
        require(len(data) == expected['bytes'] and digest(data) == expected['sha256'],
                'Input identity differs: ' + key)
        return data

    def json(self, key):
        return strict_json(self.read(key))

    def verify_all(self):
        for key in self.files:
            self.read(key)


def history_summary(history):
    require([row['update'] for row in history] == list(range(0, 200001, 500))
            and all(type(row['update']) is int for row in history), 'Incomplete/duplicate validation schedule')
    values = [row['equal_connection_force_mae_eV_per_A'] for row in history]
    require(all(finite_number(value) for value in values), 'Invalid validation score')
    best = min(range(len(values)), key=values.__getitem__)
    late = [row for row in history if 190500 <= row['update'] <= 200000]
    require(len(late) == 20, 'Fixed late20 support')
    return {'best_update': history[best]['update'], 'best': values[best], 'last': values[-1],
            'late20': statistics.mean(row['equal_connection_force_mae_eV_per_A'] for row in late),
            'late20_updates': [row['update'] for row in late]}


def checkpoint_metadata(data, provenance):
    envelope = msgpack.unpackb(data, raw=False, strict_map_key=False)
    require(envelope['provenance'] == json.dumps(provenance, sort_keys=True, separators=(',', ':')),
            'Checkpoint provenance differs')
    for value, binding in [('payload', 'sha256'), ('control_payload', 'control_sha256')]:
        require(type(envelope[value]) is bytes and digest(envelope[value]) == envelope[binding],
                'Checkpoint ' + value + ' identity differs')
    # Decode control metadata only. Model payload and embedded BEST stay bytes.
    control = msgpack.unpackb(envelope['control_payload'], raw=False, strict_map_key=False)
    require(type(control['best_state']) is bytes, 'Embedded BEST must remain opaque')
    return {'payload': envelope['payload'], 'control': control,
            'payload_sha256': envelope['sha256'], 'control_sha256': envelope['control_sha256']}


def reduce_predictions(ids, groups, energy, forces, labels):
    ids, groups = list(ids), list(groups)
    expected = list(labels['record_ids'])
    require(len(ids) == len(set(ids)) and len(expected) == len(set(expected))
            and set(ids) == set(expected), 'Prediction record identity support differs')
    require(all(isinstance(value, str) and value for value in ids), 'Invalid record ID')
    index = {name: i for i, name in enumerate(expected)}
    order = [index[name] for name in ids]
    require(groups == [labels['groups'][i] for i in order], 'Prediction connection mapping differs')
    e, f = np.asarray(energy), np.asarray(forces)
    target_e, target_f = labels['energy_residual'][order], labels['forces'][order]
    require(e.shape == target_e.shape and f.shape == target_f.shape and f.ndim == 3,
            'Prediction shape differs')
    require(e.dtype == np.float64 and f.dtype == np.float64, 'Prediction precision differs')
    require(np.isfinite(e).all() and np.isfinite(f).all(), 'Nonfinite predictions')
    difference = f - target_f
    record_errors = np.abs(difference).mean(axis=(1, 2))
    zero_errors = np.abs(target_f).mean(axis=(1, 2))
    group_array = np.asarray(groups)
    means = {group: float(record_errors[group_array == group].mean()) for group in sorted(set(groups))}
    return {'record_count': len(ids), 'connection_count': len(means),
            'per_connection_force_mae_eV_per_A': means,
            'equal_connection_force_mae_eV_per_A': statistics.mean(means.values()),
            'pooled_force_mae_eV_per_A': float(np.abs(difference).mean()),
            'pooled_component_force_rmse_eV_per_A': float(np.sqrt(np.mean(difference ** 2))),
            'energy_mae_eV': float(np.abs(e - target_e).mean()),
            'zero_force_equal_connection_mae_eV_per_A': statistics.mean(
                float(zero_errors[group_array == group].mean()) for group in means)}


def paired_summaries(rows):
    values = {row['candidate']: row for row in rows}
    require(len(values) == len(rows) and set(values) == CANDIDATES, 'Missing/duplicate candidate or seed')
    result = {}
    for family in ('SP', 'S64'):
        entry, signs = {}, []
        for metric in ('best', 'last', 'late20'):
            delta = [values[f'{family}-{seed}'][metric] - values[f'E64-{seed}'][metric] for seed in (0, 1)]
            require(all(math.isfinite(x) for x in delta), 'Invalid paired score')
            signs.extend(x for x in delta if x != 0)
            entry[metric] = {'seed_deltas': delta, 'mean_delta': statistics.mean(delta)}
        entry['interpretation'] = ('mixed' if signs and min(signs) < 0 < max(signs)
                                   else 'descriptive fixed-recipe comparison')
        entry['sign'] = 'scalar minus E64; positive means lower E64 error'
        result[family] = entry
    return result


def timing_summary(records):
    keys = ('elapsed_seconds', 'compilation_seconds', 'warmup_execution_seconds', 'validation_seconds_total')
    result = {'segment_count': len(records), 'scope': 'recorded whole-fit CPU timing; historical/current conditions differ'}
    for key in keys:
        values = [row.get(key) for row in records]
        result[key] = (sum(values) if values and all(row.get('timing_status') == 'recorded' for row in records)
                       and all(finite_number(v) for v in values) else None)
    steady = [row.get('steady_update_seconds_mean') for row in records]
    result['steady_update_seconds_mean'] = steady[0] if len(steady) == 1 and finite_number(steady[0]) else None
    result['timing_status'] = 'recorded' if result['elapsed_seconds'] is not None else 'unknown'
    return result


def array_identity(array):
    dtype = '<i8' if array.dtype.kind in 'iu' else '<f8'
    data = np.ascontiguousarray(array, dtype=dtype)
    return {'shape': list(data.shape), 'dtype': dtype, 'sha256': digest(data.tobytes())}


def load_labels(inputs, dataset):
    manifest = inputs.json(dataset['manifest'])
    without_identity = {k: v for k, v in manifest.items() if k != 'identity'}
    require(canonical(without_identity) == manifest['identity'] and manifest['split_hash'] == SPLIT,
            'Prepared manifest identity differs')
    info = manifest['partitions']['validation']
    ids = info['record_ids']
    require(len(ids) == len(set(ids)) == 974, 'Validation support differs')
    groups = [info['memberships'][name]['groups'] for name in ids]
    require(all(len(value) == 1 for value in groups) and {x[0] for x in groups} == GROUPS,
            'Validation groups differ')
    with np.load(io.BytesIO(inputs.read(dataset['validation'])), allow_pickle=False) as archive:
        require(set(archive.files) == set(info['arrays']), 'Validation array fields differ')
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        require(np.isfinite(value).all() and array_identity(value) == info['arrays'][key],
                'Validation array identity differs: ' + key)
    require(arrays['forces'].shape == (974, 21, 3), 'Validation atom count differs')
    return {'record_ids': ids, 'groups': [g[0] for g in groups], **arrays}, manifest


def verify_recipe(name, config, provenance, summary, manifest):
    family, seed_text = name.split('-')
    model = {'features': 88 if family == 'SP' else 64, 'max_degree': 1 if family == 'E64' else 0,
             'iterations': 2, 'radial_basis': 8, 'cutoff_A': 5., 'energy_readout_init': 'zero'}
    require(config['model'] == model and config['seed'] == int(seed_text), 'Architecture/seed differs: ' + name)
    for key, expected in {'arm': 'energy', 'backend': 'cpu', 'batch_size': 4, 'learning_rate': .003,
                          'max_updates': 200000, 'validation_interval': 500, 'expected_split_hash': SPLIT,
                          'mode': 'integration', 'consistency_strength': 0., 'expected_learning_subset_hash': None,
                          'loss_weights': {'energy': .014584365670390719, 'force': 1.9854156343296092}}.items():
        require(config[key] == expected, 'Recipe differs: ' + name + '/' + key)
    computation = {key: value for key, value in config.items()
                   if key not in ('data_directory', 'run_directory', 'resume_checkpoint', 'max_updates', 'wall_seconds')}
    require(canonical(computation) == provenance['config_computation_hash'], 'Config/provenance binding differs')
    require(provenance['dataset_identity'] == manifest['identity']
            and provenance['normalization'] == manifest['normalization']
            and provenance['split_hash'] == SPLIT and provenance['parameter_count'] == COUNTS[family],
            'Dataset/parameter identity differs')
    require(summary['provenance'] == provenance and summary['status'] == 'passed'
            and summary['stop_reason'] == 'max_updates' and summary['updates'] == 200000
            and summary['checkpoint_new_process_verified'] is True
            and summary['best_checkpoint_new_process_verified'] is True, 'Incomplete fit or reload evidence')


def verify_score_identity(score, config, provenance, scoring_source_sha256):
    require(score['seed'] == config['seed'] and score['model'] == config['model']
            and score['parameter_count'] == provenance['parameter_count']
            and score['source_fingerprint'] == provenance['code_fingerprint']
            and score['training_revision'] == provenance['revision']
            and score['backend'] == config['backend'] == provenance['backend']
            and score['analysis_source_sha256'] == scoring_source_sha256, 'Cached scoring recipe identity differs')


def audit_study(repository, retained_root, inventory):
    started = time.monotonic()
    inputs = BoundInputs(repository, retained_root, inventory)
    require(set(inventory['runs']) == CANDIDATES, 'Six fixed candidates required')
    inputs.verify_all()
    labels, manifest = load_labels(inputs, inventory['dataset'])
    launch = inputs.json(inventory['launch'])
    counts = inputs.json('repository:reports/evidence/2026-10-20/representation/parameter-counts.json')
    require(counts['reference'] == COUNTS['E64'] and counts['scalar_width'] == 88, 'Parameter-count record differs')
    scalar_counts = {int(width): count for width, count in counts['scalar_counts'].items()}
    require(set(scalar_counts) == set(range(8, 257)), 'Parameter-search support differs')
    chosen = min(scalar_counts, key=lambda width: (abs(scalar_counts[width]-counts['reference']), width))
    require(chosen == 88 and scalar_counts[88] == COUNTS['SP'] and scalar_counts[64] == COUNTS['S64']
            and abs(scalar_counts[88]-counts['reference'])/counts['reference'] <= .01,
            'Mechanical parameter match differs')
    assert_close(counts, inputs.json('retained:reports/execution/representation-study-2026-10-20/parameter-counts.json'),
                 'Published parameter-count record')
    rows, trajectories, connection_deltas, source_fingerprints = [], [], [], set()
    maxima = {'cached_metric_max_abs_difference': 0., 'history_max_abs_difference': 0.}
    for name in sorted(CANDIDATES):
        binding = inventory['runs'][name]
        raw_score_bytes = inputs.read(binding['raw_score'])
        require(digest(raw_score_bytes) == inputs.read(binding['score_hash']).decode().strip(), 'Cached score hash differs')
        raw_score, exported = strict_json(raw_score_bytes), inputs.json(binding['export'])
        assert_close({k: v for k, v in exported.items() if k != 'score_directory'}, raw_score, name + ' export')
        require(raw_score['status'] == 'independently_scored_validation_only', 'Unverified scoring status')
        config, provenance, summary = [inputs.json(binding[key]) for key in ('config.json', 'provenance.json', 'summary.json')]
        verify_recipe(name, config, provenance, summary, manifest)
        verify_score_identity(raw_score, config, provenance, inputs.files[inventory['scoring_source']]['sha256'])
        source_fingerprints.add((provenance['code_fingerprint'], provenance['lock_sha256'], provenance['python'],
                                 canonical(provenance['dependencies'])))
        if name.startswith('E64'):
            for filename, expected in launch['reference_sha256'][name].items():
                require(inputs.files[binding[filename]]['sha256'] == expected, 'Historical reference binding differs')
        checkpoints = {}
        for role in ('best', 'last'):
            data = inputs.read(binding[role + '.msgpack'])
            require(digest(data) == raw_score['checkpoint_sha256'][role], 'Checkpoint/score binding differs')
            checkpoints[role] = checkpoint_metadata(data, provenance)
        control = checkpoints['last']['control']
        history = control['validation_history']
        require(history == inputs.json(binding['validation.json']), 'Standalone history differs from final control')
        metrics = history_summary(history)
        require(control['best_update'] == summary['best_update'] == metrics['best_update'], 'Selected BEST update differs')
        assert_close(control['best_score'], metrics['best'], 'control BEST')
        require(checkpoints['best']['payload'] == control['best_state'], 'Embedded BEST payload differs')
        prefix = checkpoints['best']['control']['validation_history']
        require(prefix == history[:metrics['best_update']//500 + 1], 'BEST control prefix differs')
        for role in ('best', 'last'):
            proof_key = 'best-checkpoint-verification.json' if role == 'best' else 'checkpoint-verification.json'
            proof = inputs.json(binding[proof_key])
            require(proof['status'] == 'passed' and proof['role'] == role
                    and proof['update'] == (metrics['best_update'] if role == 'best' else 200000)
                    and proof['backend'] == 'cpu' and finite_number(proof['maximum_prediction_difference'])
                    and proof['maximum_prediction_difference'] <= proof['tolerance'] == 1e-10,
                    'Historical reload proof differs')
        for key in ('best', 'last', 'late20'):
            cached = raw_score['history'][key + '_equal_connection_force_mae_eV_per_A']
            assert_close(metrics[key], cached, name + ' history ' + key)
            maxima['history_max_abs_difference'] = max(maxima['history_max_abs_difference'], abs(metrics[key] - cached))
        require(metrics['best_update'] == raw_score['history']['best_update']
                and metrics['late20_updates'] == raw_score['history']['late20_updates'], 'Cached history support differs')
        endpoint = {}
        for role in ('best', 'last'):
            data = inputs.read(binding[role + '_predictions'])
            require(digest(data) == raw_score['prediction_files'][role], 'Cached prediction hash differs')
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                require(set(archive.files) == {'record_ids', 'groups', 'energy', 'forces'}, 'Cached prediction fields differ')
                endpoint[role] = reduce_predictions(archive['record_ids'].tolist(), archive['groups'].tolist(),
                                                    archive['energy'], archive['forces'], labels)
            assert_close(endpoint[role], raw_score['roles'][role], name + ' ' + role)
            assert_close(endpoint[role]['equal_connection_force_mae_eV_per_A'], metrics[role], name + ' selected history')
            for key, value in endpoint[role].items():
                if type(value) is float:
                    maxima['cached_metric_max_abs_difference'] = max(maxima['cached_metric_max_abs_difference'],
                        abs(value - raw_score['roles'][role][key]))
        require(len(raw_score['segment_timing']) == 1, 'Unexpected segment count in fixed completed study')
        recorded = raw_score['segment_timing'][0]
        timing_keys = ('elapsed_seconds', 'compilation_seconds', 'warmup_execution_seconds',
                       'validation_seconds_total', 'steady_update_seconds_mean')
        for key in timing_keys:
            if finite_number(recorded.get(key)) and finite_number(summary.get(key)):
                assert_close(recorded[key], summary[key], 'Recorded timing ' + key)
        timing = timing_summary([{'timing_status': 'recorded', **{key: summary.get(key) for key in timing_keys}}])
        assert_close(summary['elapsed_seconds'], raw_score['final_segment_elapsed_seconds'], 'Saved timing')
        row = {'candidate': name, 'family': name.split('-')[0], 'seed': config['seed'],
               'parameter_count': COUNTS[name.split('-')[0]], 'reused': name.startswith('E64'),
               **metrics, 'roles': endpoint, 'timing': timing, 'training_revision': provenance['revision'],
               'source_fingerprint': provenance['code_fingerprint']}
        rows.append(row)
        trajectories.extend({'candidate': name, 'seed': row['seed'], 'update': h['update'],
                             'force_mae_meV_per_A': h['equal_connection_force_mae_eV_per_A'] * 1000} for h in history)
        del checkpoints, history, prefix, control
    require(len(source_fingerprints) == 1, 'Numerical source/environment differs across candidates')
    pairs = paired_summaries(rows)
    published_pairs = inputs.json('repository:reports/evidence/2026-10-20/representation/paired.json')['pairs']
    for family in ('SP', 'S64'):
        for metric in ('best', 'late20'):
            assert_close(pairs[family][metric]['seed_deltas'], published_pairs[family][metric + '_seed_deltas'], 'Published pair')
            assert_close(pairs[family][metric]['mean_delta'], published_pairs[family][metric + '_mean_delta'], 'Published pair mean')
    values = {row['candidate']: row for row in rows}
    for family in ('SP', 'S64'):
        for seed in (0, 1):
            for role in ('best', 'last'):
                for group in sorted(GROUPS):
                    a = values[f'{family}-{seed}']['roles'][role]['per_connection_force_mae_eV_per_A'][group]
                    b = values[f'E64-{seed}']['roles'][role]['per_connection_force_mae_eV_per_A'][group]
                    connection_deltas.append({'contrast': family + ' minus E64', 'seed': seed, 'role': role,
                                              'connection': group, 'delta_meV_per_A': (a - b) * 1000})
    families = {}
    for family in ('E64', 'SP', 'S64'):
        selected = [values[f'{family}-{seed}'] for seed in (0, 1)]
        families[family] = {metric: {'mean': statistics.mean(row[metric] for row in selected),
                                     'seed_values': [row[metric] for row in selected]}
                            for metric in ('best', 'last', 'late20')}
        elapsed = [row['timing']['elapsed_seconds'] for row in selected]
        families[family]['mean_recorded_fit_seconds'] = statistics.mean(elapsed) if all(x is not None for x in elapsed) else None
        families[family]['parameter_count'] = COUNTS[family]
    inputs.verify_all()
    return {'schema': 1, 'status': 'VERIFIED_SAVED_VALIDATION_ANALYSIS', 'rows': rows, 'families': families,
            'pairs': pairs, 'connection_deltas': connection_deltas, 'trajectories': trajectories,
            'verification': {'candidate_count': 6, 'endpoint_prediction_files': 12, 'history_events': len(trajectories),
                'cached_force_components': 12*974*21*3, 'historical_reload_proofs': 12,
                'input_file_count': len(inputs.files), 'inputs_unchanged': True,
                'reduction_atol': ATOL, 'reduction_rtol': RTOL, **maxima},
            'source': {'analysis_sha256': digest(Path(__file__).read_bytes()),
                       'commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()},
            'input_inventory': inventory,
            'limits': ['Two seeds and five reused validation connections; descriptive comparison only',
                'Recipe selected on E64, without scalar retuning; BEST selection is optimistic',
                'Scalar energy-derived forces remain equivariant; no causal symmetry claim',
                'Historical/current CPU timings are observational, not controlled efficiency or inference speed',
                'No model payload restored, learned prediction, fitting, recovery or held-out test payload'],
            'analysis_elapsed_seconds': time.monotonic() - started}


def write_artifacts(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    require(result.get('status') == 'VERIFIED_SAVED_VALIDATION_ANALYSIS', 'Cannot export unverified interpretation')
    (output/'analysis.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    def csv_file(name, records):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    csv_file('run_metrics.csv', [{'candidate': row['candidate'], 'seed': row['seed'], 'parameters': row['parameter_count'],
        'best_update': row['best_update'], **{metric+'_meV_per_A': row[metric]*1000 for metric in ('best','last','late20')},
        'recorded_fit_seconds': row['timing']['elapsed_seconds'],
        'steady_update_seconds_mean': row['timing']['steady_update_seconds_mean']} for row in result['rows']])
    csv_file('paired_seed_deltas.csv', [{'contrast': family+' minus E64','seed': seed,'summary': metric,
        'delta_meV_per_A': values[metric]['seed_deltas'][seed]*1000}
        for family, values in result['pairs'].items() for metric in ('best','last','late20') for seed in (0,1)])
    csv_file('connection_deltas.csv', result['connection_deltas'])
    csv_file('validation_trajectories.csv', result['trajectories'])
    csv_file('endpoint_metrics.csv', [{'candidate': row['candidate'], 'role': role,
        **{key: value for key, value in metrics.items() if key != 'per_connection_force_mae_eV_per_A'}}
        for row in result['rows'] for role, metrics in row['roles'].items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'retained-root', 'inventory', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output already exists; preserve it')
    try:
        result = audit_study(args.repository, args.retained_root, strict_json(args.inventory.read_bytes()))
        write_artifacts(result, args.output)
        print(json.dumps({'status': result['status'], **result['verification']}, sort_keys=True))
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'failure.json').write_text(json.dumps({'status':'FAILED_SAVED_DATA_AUDIT',
            'exception_type':type(exc).__name__, 'reason':str(exc), 'interpretation_accepted':False}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
