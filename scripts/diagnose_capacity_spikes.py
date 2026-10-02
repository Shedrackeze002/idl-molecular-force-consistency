"""Bounded immutable capacity checkpoint analysis; numerical imports are guarded."""
import argparse
import hashlib
import json
from pathlib import Path
import time

CONFIG_ID = '0ac07e84eb570ba926d2cc3f7338f120c78c0fc6dd8ea18174bcfc0f86b2ab1a'
EXECUTION_RECORDS = Path('reports/execution/spike-diagnostic-2026-10-02')
LAUNCH_INPUTS = {
    'scripts/diagnose_capacity_spikes.py', 'tests/test_capacity_spikes.py', 'configs/aspirin-capacity-spike.json',
    'docs/superpowers/specs/2026-10-02-capacity-spike-design.md', 'docs/superpowers/plans/2026-10-02-capacity-spike.md',
    'docs/reviews/2026-10-02-capacity-spike-scientific-plan-followup.md',
    'docs/reviews/2026-10-02-capacity-spike-engineering-plan-followup.md',
    'docs/reviews/2026-10-02-capacity-spike-implementation.md',
    str(EXECUTION_RECORDS / 'source-pins.json'), str(EXECUTION_RECORDS / 'focused-verification.txt'),
    str(EXECUTION_RECORDS / 'full-suite.txt'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from force_consistency.width_rate_replacement import atomic
    atomic(Path(path), value)


def verify_inventory(root, inventory, identity):
    from force_consistency.capacity_contract import no_symlinks
    root = no_symlinks(root); inventory = no_symlinks(inventory)
    if digest(inventory) != identity:
        raise ValueError('Source inventory identity differs')
    pins = read(inventory)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if set(pins) != actual:
        raise ValueError('Source inventory members differ')
    for name, expected in pins.items():
        path = no_symlinks(root / name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Source file hash differs: ' + name)
    return len(pins)


def residual_summary(energy_error, force_error, groups):
    import numpy as np
    e = np.asarray(energy_error, dtype=float); f = np.asarray(force_error, dtype=float)
    if (e.ndim != 1 or not e.size or f.ndim != 3 or f.shape[0] != len(e)
            or f.shape[-1] != 3 or not f.size or len(groups) != len(e)
            or not np.isfinite(e).all() or not np.isfinite(f).all()):
        raise ValueError('Invalid residual inputs')

    def pool(x, y):
        mean = float(np.mean(x)); mse = float(np.mean(x * x))
        variance = float(np.mean((x - mean) ** 2))
        return {'signed_mean_energy_error_eV': mean, 'mean_energy_error_squared_eV2': mean ** 2,
                'centered_energy_error_variance_eV2': variance, 'energy_mse_eV2': mse,
                'force_mse_eV2_per_A2': float(np.mean(y * y)), 'record_count': len(x),
                'decomposition_error_eV2': mse - mean ** 2 - variance,
                'offset_share': mean ** 2 / mse if mse else None,
                'offset_share_unavailable_reason': None if mse else 'zero_energy_mse'}

    result = pool(e, f)
    per = {g: pool(e[[i for i, v in enumerate(groups) if v == g]],
                   f[[i for i, v in enumerate(groups) if v == g]]) for g in sorted(set(groups))}
    result['per_group'] = per
    result['equal_group'] = {k: float(np.mean([r[k] for r in per.values()]))
                             for k in ('energy_mse_eV2', 'force_mse_eV2_per_A2')}
    return result


def residual_change(before, after):
    delta = after['energy_mse_eV2'] - before['energy_mse_eV2']
    mean2 = after['mean_energy_error_squared_eV2'] - before['mean_energy_error_squared_eV2']
    var = after['centered_energy_error_variance_eV2'] - before['centered_energy_error_variance_eV2']
    return {'energy_mse_change_eV2': delta, 'mean_squared_change_eV2': mean2,
            'variance_change_eV2': var, 'identity_error_eV2': delta - mean2 - var,
            'offset_increase_share': mean2 / delta if delta > 0 else None,
            'offset_increase_share_unavailable_reason': None if delta > 0 else 'nonpositive_mse_increase'}


def bias_check(errors, atoms, scale, observed_energy, observed_force, weight):
    import numpy as np
    expected = float(2 * np.mean(np.asarray(atoms) * np.asarray(errors)) / scale ** 2)
    return {'expected_unweighted_energy': expected, 'observed_unweighted_energy': float(observed_energy),
            'expected_weighted_energy': weight * expected, 'observed_weighted_energy': weight * float(observed_energy),
            'observed_force': float(observed_force),
            'valid': bool(np.isclose(expected, observed_energy, atol=1e-10, rtol=1e-8)
                          and abs(observed_force) <= 1e-12)}


def gradient_metrics(ge, gf, joint, weighted_joint, weights):
    import jax
    import numpy as np
    from force_consistency.gradient_diagnostics import gradient_geometry
    a, b = weights['energy'], weights['force']
    def identity(actual, eweight, fweight):
        wanted = jax.tree.map(lambda e, f: eweight * e + fweight * f, ge, gf)
        if jax.tree.structure(actual) != jax.tree.structure(wanted):
            raise ValueError('Joint derivative schema differs')
        errors = [np.asarray(x) - np.asarray(y) for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True)]
        return {'valid': all(np.allclose(x, y, atol=1e-10, rtol=1e-8) for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True)),
                'maximum_absolute_error': max(float(np.max(np.abs(x))) for x in errors)}
    plain = gradient_geometry(ge, gf)
    weighted = gradient_geometry(jax.tree.map(lambda x: a * x, ge), jax.tree.map(lambda x: b * x, gf))
    def relabel(row, weighted_components=False):
        if 'energy_norm' not in row:
            for value in row.values(): relabel(value, weighted_components)
            return
        olde = row.pop('energy_raw_joint_descent_slope'); oldf = row.pop('force_raw_joint_descent_slope')
        prefix = 'weighted_component_negative_weighted_joint' if weighted_components else 'scaled_loss_negative_unweighted_joint'
        row['energy_' + prefix + '_slope'] = olde
        row['force_' + prefix + '_slope'] = oldf
        if not weighted_components:
            row['energy_scaled_loss_negative_weighted_joint_slope'] = -(a * row['energy_norm'] ** 2 + b * row['dot'])
            row['force_scaled_loss_negative_weighted_joint_slope'] = -(a * row['dot'] + b * row['force_norm'] ** 2)
        row['cosine_unavailable_reason'] = 'zero_gradient_norm' if row['cosine'] is None else None
        row['force_over_energy_norm_unavailable_reason'] = 'zero_energy_gradient_norm' if row['force_over_energy_norm'] is None else None
    relabel(plain); relabel(weighted, True)
    return {'unweighted': plain, 'weighted': weighted,
            'unweighted_joint_identity': identity(joint, 1., 1.),
            'weighted_joint_identity': identity(weighted_joint, a, b),
            'scope': 'Scaled objective Euclidean geometry; per-block values are block contributions. No raw-MSE or actual Adam direction.'}


class Journal:
    def __init__(self, root, check, *, ready=None, finite=None):
        self.root = Path(root); self.check = check
        if ready is None:
            import jax
            ready = jax.block_until_ready
        if finite is None:
            from force_consistency.runs import assert_finite
            finite = assert_finite
        self.ready = ready; self.finite = finite
        self.context = None
        self.work = {'counts': {}, 'in_flight': None, 'training_updates': 0,
                     'random_parameter_initializations': 0, 'optimizer_proposals': 0}
        self.save()

    def save(self):
        write(self.root / 'work.json', self.work)

    def call(self, kind, count, fn, *args, on_returned=lambda value: None):
        self.check()
        row = self.work['counts'].setdefault(kind, {'attempted_calls': 0, 'returned_calls': 0,
                'returned_structure_exposures': 0, 'valid_calls': 0, 'valid_structure_exposures': 0})
        row['attempted_calls'] += 1
        self.work['in_flight'] = {'kind': kind, 'structure_count': count, 'completion': 'unknown_until_ready', 'context': self.context}
        self.save()
        result = fn(*args); self.ready(result)
        row['returned_calls'] += 1; row['returned_structure_exposures'] += count
        self.work['in_flight'] = None; self.save()
        on_returned(result)
        self.finite(result)
        row['valid_calls'] += 1; row['valid_structure_exposures'] += count; self.save()
        return result


def finite_differences(params, ge, gf, evaluate, progress, *, seed, context=lambda d: None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    result = {}; rng = np.random.default_rng(seed)
    for i, (name, grad) in enumerate([('energy', ge), ('force', gf)]):
        norm = np.sqrt(sum(float(np.vdot(x, x)) for x in jax.tree.leaves(grad)))
        if norm > 1e-12:
            direction = jax.tree.map(lambda x: x / norm, grad); kind = 'unit_gradient'
        else:
            raw = jax.tree.map(lambda x: jnp.asarray(rng.choice([-1., 1.], size=x.shape)), params)
            count = sum(x.size for x in jax.tree.leaves(raw))
            direction = jax.tree.map(lambda x: x / np.sqrt(count), raw); kind = 'seeded_rademacher'
        autodiff = sum(float(np.vdot(g, d)) for g, d in zip(jax.tree.leaves(grad), jax.tree.leaves(direction), strict=True))
        records = []; result[name] = {'direction_kind': kind, 'autodiff_directional_derivative': autodiff, 'steps': records, 'status': 'running'}
        for step in (1e-4, 1e-5):
            plus = jax.tree.map(lambda p, d: p + step * d, params, direction)
            minus = jax.tree.map(lambda p, d: p - step * d, params, direction)
            context({'objective': name, 'step': step, 'sign': 'plus'}); vp = evaluate(plus)
            context({'objective': name, 'step': step, 'sign': 'minus'}); vm = evaluate(minus)
            fd = float((vp[i] - vm[i]) / (2 * step))
            records.append({'step': step, 'finite_difference': fd, 'absolute_error': abs(fd - autodiff),
                            'valid': bool(np.isclose(fd, autodiff, atol=1e-8, rtol=1e-4))})
            progress(result)
        result[name]['valid'] = any(r['valid'] for r in records); result[name]['status'] = 'completed'
    result['valid'] = all(result[k]['valid'] for k in ('energy', 'force'))
    return result


def reserve_run(root):
    from force_consistency.capacity_contract import no_symlinks
    root = no_symlinks(root); root.mkdir(exist_ok=False)
    with (root / 'attempt.json').open('x') as f:
        json.dump({'ordinal': 1, 'maximum_invocations': 1}, f)


def execute_cases(root, cases, analyzer, check):
    root = Path(root); results = {}; blocked = None
    for case in cases:
        path = root / case['name']; path.mkdir(exist_ok=False)
        started = time.perf_counter()
        summary = {'case': case, 'status': 'blocked', 'verification_valid': False, 'executed': False}
        if blocked is not None:
            summary['stop_reason'] = blocked
        else:
            write(path / 'prospective.json', case)
            try:
                check(); summary['executed'] = True
                observed = analyzer(case, path)
                write(path / 'observations.json', observed)
                if not observed['verification_valid']: raise ValueError('Case verification failed')
                check()
                summary.update(status='completed', verification_valid=True)
            except Exception as exc:
                summary.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
                blocked = case['name'] + ': ' + summary['failure_reason']
        summary['elapsed_seconds'] = time.perf_counter() - started
        write(path / 'summary.json', summary); results[case['name']] = summary
        write(root / 'cases.json', results)
        print(json.dumps({'case': case['name'], 'status': summary['status'], 'failure_reason': summary.get('failure_reason')}), flush=True)
    return results


def saved_cases(root, cases, *, source_integrity):
    root = Path(root); result = {}
    for case in cases:
        path = root / case['name']
        interrupted = (path / 'work.json').exists() or (path / 'partial.json').exists()
        summary = read(path / 'summary.json') if (path / 'summary.json').exists() else {'status': 'interrupted' if interrupted else 'blocked', 'verification_valid': False}
        observed = read(path / 'observations.json') if (path / 'observations.json').exists() else None
        endorsed = bool(source_integrity and summary.get('status') == 'completed' and summary.get('verification_valid')
                        and observed and observed.get('verification_valid'))
        result[case['name']] = {'summary': summary, 'endorsed': endorsed, 'observations': observed if endorsed else None,
                              'raw_observations_exist': observed is not None,
                              'work': read(path / 'work.json') if (path / 'work.json').exists() else None}
    return result


def verify_inputs(config):
    from force_consistency import capacity_contract as cc, capacity_report as cr
    if cc.canonical_hash(config) != CONFIG_ID:
        raise ValueError('Unreviewed spike configuration')
    verify_inventory(config['source_root'], config['source_inventory'], config['source_inventory_sha256'])
    if digest(config['source_pins']) != config['source_pins_sha256']:
        raise ValueError('Diagnostic source pin identity differs')
    pins = read(config['source_pins']); inventory = read(config['source_inventory'])
    extras = {config['capacity_config'], 'reports/capacity-diagnostics.json', 'reports/capacity-diagnostics.md',
              'reports/execution/capacity-2026-10-02/source-pins.json'}
    if set(pins) != {config['source_root'] + '/' + name for name in inventory} | extras:
        raise ValueError('Incomplete diagnostic source pin members')
    cc.verify_pins(pins)
    cc.verify_sources(read(config['capacity_config']))
    cc.verify_code(read(Path(config['source_root']) / 'code.json'))
    report = cr.build_report(config['source_root']); cr.verify_saved_report(report)
    if not report['execution_valid'] or digest('reports/capacity-diagnostics.json') != config['capacity_report_sha256']:
        raise ValueError('Source capacity report invalid')
    return report


def load_snapshot(root, update, provenance):
    from force_consistency import capacity_contract as cc
    root = Path(root); history = read(root / f'histories/{update}.json')
    row = history[-1]
    if row['update'] != update: raise ValueError('Historical selected update differs')
    receipt = {'update': update, 'checkpoint': f'snapshots/{update}.msgpack',
               'checkpoint_sha256': digest(root / f'snapshots/{update}.msgpack'), 'state_hashes': row['state_hashes']}
    metadata = cc.audit_state(root, receipt, provenance)
    # No initializer or optimizer construction: decode the already audited arrays.
    from flax import serialization
    from force_consistency.runs import _read_checkpoint, assert_finite, assert_parameter_dtype
    from force_consistency.late_rate import state_hashes
    state = serialization.msgpack_restore(_read_checkpoint(root / receipt['checkpoint'], provenance)['payload'])
    assert_finite(state); assert_parameter_dtype(state['params'])
    if state_hashes(state) != metadata['state_hashes']:
        raise ValueError('Decoded parameter/optimizer hashes differ')
    return state


def compare_scores(actual, expected):
    import numpy as np
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise ValueError('Score schema differs')
        for k in expected: compare_scores(actual[k], expected[k])
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, atol=1e-10, rtol=1e-8): raise ValueError('Historical score differs')
    elif actual != expected:
        raise ValueError('Historical score metadata differs')


def analyze_case(state, parts, model, norm, weights, history_row, journal, on_partial, endpoint_reference=None):
    import jax
    import numpy as np
    from flax import serialization
    from force_consistency import coverage_runs as cov, gradient_diagnostics as gd, loss_balance as lb
    from force_consistency.metrics import partition_batch
    from force_consistency.runs import assert_finite
    from force_consistency.late_rate import state_hashes
    def fullhash(): return hashlib.sha256(serialization.to_bytes(jax.tree.map(lambda x: x, state))).hexdigest()
    before = fullhash(); before_params = state_hashes(state)
    observed = {'update': int(state['update']), 'state_full_hash_before': before,
                'state_hashes_before': before_params, 'residuals': {}, 'historical_score_match': {},
                'original_baseline_fractions': {},
                'record_ids': {r: list(p.record_ids) for r, p in parts.items()}}
    arrays = {}; params = state['params']; size = 16
    def save(): on_partial(observed)
    try:
        for role, part in parts.items():
            ee = []; ff = []
            for i in range(0, len(part.record_ids), size):
                ix = np.arange(i, min(i + size, len(part.record_ids))); batch = partition_batch(part, ix)
                batch_dir = journal.root / 'prediction-batches'; batch_dir.mkdir(exist_ok=True)
                npz = batch_dir / f'{role}-{i:06d}.npz'; meta = npz.with_suffix('.json')
                record_ids = [part.record_ids[j] for j in ix]
                journal.context = {'role': role, 'first_index': i, 'record_ids': record_ids}
                def retain_prediction(returned):
                    e, f = returned
                    with npz.open('xb') as file:
                        np.savez(file, energy_prediction=np.asarray(e), force_prediction=np.asarray(f),
                                energy_target=part.energy_residual[ix], force_target=part.forces[ix],
                                record_ids=np.asarray(record_ids))
                    write(meta, {'status': 'ready_returned_unverified', 'record_ids': record_ids,
                                 'state_full_hash': before, 'role': role, 'npz_sha256': digest(npz)})
                e, f = journal.call('prediction', len(ix), cov._predict(model), params, batch, on_returned=retain_prediction)
                record = read(meta); record['status'] = 'finite_valid'; write(meta, record)
                ee.append(np.asarray(e)); ff.append(np.asarray(f).reshape(len(ix), *part.forces.shape[1:]))
            e = np.concatenate(ee); f = np.concatenate(ff)
            compare_scores(cov._group_score(part, e, f), history_row['metrics'][role])
            error = e - part.energy_residual; ferr = f - part.forces
            groups = []
            for rid in part.record_ids:
                if len(part.memberships[rid]['groups']) != 1: raise ValueError('Ambiguous group membership')
                groups.append(part.memberships[rid]['groups'][0])
            residual = residual_summary(error, ferr, groups)
            if abs(residual['decomposition_error_eV2']) > 1e-10 + 1e-8 * residual['energy_mse_eV2']:
                raise ValueError('Residual decomposition differs')
            observed['residuals'][role] = residual; observed['historical_score_match'][role] = True
            observed['original_baseline_fractions'][role] = history_row.get(role + '_gate', {}).get('fraction_remaining')
            arrays.update({role + '_energy': e, role + '_forces': f, role + '_energy_error': error,
                           role + '_record_force_mse': np.mean(ferr ** 2, axis=(1, 2))})
            np.savez(journal.root / 'predictions.npz', **arrays); save()
        if endpoint_reference is not None:
            maximum = 0.
            with np.load(endpoint_reference, allow_pickle=False) as ref:
                for name in ('fitting_energy', 'fitting_forces', 'anchor_energy', 'anchor_forces'):
                    if arrays[name].shape != ref[name].shape or not np.allclose(arrays[name], ref[name], atol=1e-10, rtol=0):
                        raise ValueError('Endpoint reference prediction mismatch')
                    maximum = max(maximum, float(np.max(np.abs(arrays[name] - ref[name]))))
            observed['endpoint_reference_max_error'] = maximum; save()
        fitting = parts['fitting']; total = len(fitting.record_ids)
        batches = [partition_batch(fitting, np.arange(i, min(i + size, total))) for i in range(0, total, size)]
        values, split = gd._objectives(model, norm)
        partial_gradients = {}
        def accumulate(fn, label, nested=False):
            value = 0.; gradient = None
            journal.context = {'objective': label}
            for batch in batches:
                returned = journal.call('gradient', len(batch['energy']), fn, params, batch)
                loss, g = (returned[0][0], returned[1]) if nested else returned
                fraction = len(batch['energy']) / total; value += float(loss) * fraction
                weighted = jax.tree.map(lambda x: x * fraction, g)
                gradient = weighted if gradient is None else jax.tree.map(lambda x, y: x + y, gradient, weighted)
                partial_gradients[label] = gradient
                (journal.root / 'partial-gradients.msgpack').write_bytes(serialization.msgpack_serialize(
                        jax.tree.map(lambda x: np.asarray(x), partial_gradients)))
                observed.setdefault('partial_scaled_losses', {})[label] = value; save()
            return value, gradient
        le, ge = accumulate(split[0], 'energy_scaled'); lf, gf = accumulate(split[1], 'force_scaled')
        lj, gj = accumulate(cov._gradient(model, norm), 'joint_unweighted', True)
        lw, gw = accumulate(lb._weighted_gradient(model, norm, (weights['energy'], weights['force'])), 'joint_weighted', True)
        observed['losses'] = {'energy_scaled': le, 'force_scaled': lf, 'unweighted_joint': lj, 'weighted_joint': lw}
        observed['loss_identity_valid'] = bool(np.isclose(lj, le + lf, atol=1e-10, rtol=1e-8)
                and np.isclose(lw, weights['energy'] * le + weights['force'] * lf, atol=1e-10, rtol=1e-8))
        observed['gradients'] = gradient_metrics(ge, gf, gj, gw, weights)
        observed['bias_check'] = bias_check(arrays['fitting_energy_error'], np.full(total, fitting.positions.shape[1]),
                norm.energy_loss_scale_eV, float(np.asarray(ge['energy_head']['0+']['bias']).ravel()[0]),
                float(np.asarray(gf['energy_head']['0+']['bias']).ravel()[0]), weights['energy'])
        (journal.root / 'gradients.msgpack').write_bytes(serialization.msgpack_serialize(
                jax.tree.map(lambda x: np.asarray(x), {'energy_scaled': ge, 'force_scaled': gf, 'joint_unweighted': gj, 'joint_weighted': gw})))
        save()
        def evaluate(pp):
            result = np.zeros(2)
            for batch in batches:
                v = journal.call('finite_difference', len(batch['energy']), values, pp, batch)
                result += np.asarray(v) * len(batch['energy']) / total
                write(journal.root / 'partial-fd-value.json', {'context': journal.context,
                        'partial_vector_value': result.tolist(), 'scale': 'Sum weighted by microbatch_count/total; partial until full support'})
            return result
        def progress(fd): observed['finite_differences'] = fd; save()
        def context(detail): journal.context = detail; write(journal.root / 'fd-context.json', detail)
        observed['finite_differences'] = finite_differences(params, ge, gf, evaluate, progress, seed=20261002, context=context)
        observed['verification_valid'] = bool(observed['loss_identity_valid'] and observed['bias_check']['valid']
                and observed['gradients']['unweighted_joint_identity']['valid']
                and observed['gradients']['weighted_joint_identity']['valid'] and observed['finite_differences']['valid'])
        assert_finite(observed)
    finally:
        observed['state_full_hash_after'] = fullhash(); observed['state_hashes_after'] = state_hashes(state)
        observed['state_unchanged'] = before == observed['state_full_hash_after'] and before_params == observed['state_hashes_after']
        if not observed['state_unchanged']: observed['verification_valid'] = False
        write(journal.root / 'state-audit.json', {k: v for k, v in observed.items() if k.startswith('state_')})
        save()
    return observed


def verify_gate(config):
    from force_consistency.capacity_contract import verify_pins
    gate = read(config['prelaunch_gate'])
    if gate.get('status') != 'verified' or set(gate.get('artifacts', {})) != LAUNCH_INPUTS:
        raise ValueError('Incomplete prelaunch artifact set')
    verify_pins(gate['artifacts'])
    return gate


def verify_execution_integrity(config, *, expected_gate=None):
    verify_inputs(config)
    actual = verify_gate(config)
    if expected_gate is not None and actual != expected_gate:
        raise ValueError('Frozen analysis launch gate replaced')
    return True


def evidence(config, identity, title, details):
    with Path(config['evidence_log']).open('a') as f:
        f.write('\n\n## ' + identity + ' — ' + title + '\n\nFrozen restored-state diagnostic; zero training updates.\n\n```json\n')
        f.write(json.dumps(details, indent=2, sort_keys=True, allow_nan=False) + '\n```\n')


def execute(config):
    from force_consistency import capacity_contract as cc
    if cc.canonical_hash(config) != CONFIG_ID: raise ValueError('Unreviewed spike configuration')
    root = Path(config['run_directory']); reserve_run(root); start = time.perf_counter()
    write(root / 'budget-start.json', {'monotonic_start': start, 'total_wall_seconds': config['wall_seconds'],
                                     'output_reserve_in_total_seconds': config['output_reserve_seconds']})
    write(root / 'config.json', config)
    summary = {'status': 'failed', 'execution_valid': False, 'source_integrity': False, 'cases': {},
               'training_updates': 0, 'random_parameter_initializations': 0, 'optimizer_proposals': 0,
               'model_definitions_constructed': 0, 'restored_states': 0, 'full_pilot': 'pending'}
    def check():
        cc.no_symlinks(root)
        if time.perf_counter() - start + config['output_reserve_seconds'] >= config['wall_seconds']:
            raise TimeoutError('Total diagnostic cap/reserve exhausted')
    evidence(config, 'SPIKE-EXECUTION-PRE-001', 'ordinal1 restored-state diagnostic reserved',
             {'config': config, 'analysis_script_sha256': digest(__file__), 'selection': 'posthoc saved maxima and neighbors; no training intervention'})
    gate = None
    try:
        check(); gate = verify_gate(config); verify_inputs(config); check()
        write(root / 'prelaunch-gate.json', gate)
        import jax
        from force_consistency import late_rate as lr
        from force_consistency.runs import code_provenance, require_backend
        write(root / 'analysis-code.json', {**code_provenance(Path.cwd()), 'script_sha256': digest(__file__),
                                           'historical_revision': read(Path(config['source_root']) / 'code.json')['revision']})
        registry = {}
        def analyzer(case, path):
            check(); case_root = Path(config['source_root']) / case['width']; prov = read(case_root / 'provenance.json')
            state = load_snapshot(case_root, case['update'], prov); summary['restored_states'] += 1
            if case['width'] not in registry:
                registry[case['width']] = (lr._parts(case_root, prov), lr.EnergyModel(lr.ModelConfig(**prov['model'])),
                                          lr.Normalization(**prov['normalization']))
                summary['model_definitions_constructed'] += 1
            parts, model, norm = registry[case['width']]
            journal = Journal(path, check)
            source_cp = case_root / f"snapshots/{case['update']}.msgpack"; source_digest = digest(source_cp)
            history_row = read(case_root / f"histories/{case['update']}.json")[-1]
            try:
                result = analyze_case(state, parts, model, norm, read(config['capacity_config'])['loss_weights'],
                        history_row, journal, lambda d: write(path / 'partial.json', d),
                        case_root / 'reference.npz' if case['update'] == 6000 else None)
                write(path / 'observations.json', result)
                return result
            finally:
                unchanged = digest(source_cp) == source_digest
                write(path / 'source-checkpoint-audit.json', {'unchanged': unchanged, 'sha256': source_digest})
                if not unchanged: raise ValueError('Source checkpoint changed')
        with jax.default_device(require_backend('cpu')):
            summary['cases'] = execute_cases(root, config['cases'], analyzer, check)
        verify_execution_integrity(config, expected_gate=gate); summary['source_integrity'] = True
        summary['execution_valid'] = all(v['status'] == 'completed' for v in summary['cases'].values())
        summary['status'] = 'completed' if summary['execution_valid'] else 'failed'
    except Exception as exc:
        summary['failure_reason'] = f'{type(exc).__name__}: {exc}'
        if (root / 'cases.json').exists(): summary['cases'] = read(root / 'cases.json')
        for case in config['cases']:
            if case['name'] not in summary['cases']:
                path = root / case['name']; path.mkdir(exist_ok=True)
                item = {'case': case, 'status': 'blocked', 'verification_valid': False, 'executed': False,
                        'stop_reason': summary['failure_reason']}
                write(path / 'summary.json', item); summary['cases'][case['name']] = item
    # Integrity audit on both successful and failed numerical paths.
    try: verify_execution_integrity(config, expected_gate=gate); summary['source_integrity'] = True
    except Exception as exc:
        summary.update(source_integrity=False, execution_valid=False, status='failed', source_integrity_failure=str(exc))
    summary['elapsed_seconds'] = time.perf_counter() - start
    summary['within_total_cap'] = summary['elapsed_seconds'] < config['wall_seconds']
    if not summary['within_total_cap']: summary.update(status='failed', execution_valid=False, failure_reason='Total diagnostic cap exceeded')
    write(root / 'cases.json', summary['cases']); write(root / 'summary.json', summary)
    evidence(config, 'SPIKE-EXECUTION-RESULT-001', 'actual bounded diagnostic outcome', summary)
    return summary


def render(root, output):
    root = Path(root); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = read(root / 'config.json')
    try:
        frozen_gate = read(root / 'prelaunch-gate.json')
        verify_execution_integrity(config, expected_gate=frozen_gate); fresh_integrity = True; integrity_failure = None
    except Exception as exc: fresh_integrity = False; integrity_failure = f'{type(exc).__name__}: {exc}'
    if (root / 'summary.json').exists():
        summary = read(root / 'summary.json')
        summary['source_integrity'] = bool(summary['source_integrity'] and fresh_integrity)
        if not summary['source_integrity'] and summary['execution_valid']:
            summary.update(status='invalid_saved_integrity', execution_valid=False)
    else:
        summary = {'status': 'interrupted', 'execution_valid': False, 'source_integrity': fresh_integrity,
                   'reconstructed_saved_only': True, 'elapsed_seconds': None, 'within_total_cap': None,
                   'stop_reason': 'Missing immutable completion summary; no numerical retry', 'full_pilot': 'pending'}
    summary['fresh_integrity_failure'] = integrity_failure
    cases = saved_cases(root, config['cases'], source_integrity=summary['source_integrity'])
    contrasts = {}
    if summary['execution_valid']:
        for width, updates in [('width32', [3600, 3700, 3800]), ('width64', [4900, 5000, 5100])]:
            selected = [cases.get(f'{width}-u{u}') for u in updates]
            if all(c and c['endorsed'] for c in selected):
                contrasts[width] = {role: {'pre_to_peak': residual_change(selected[0]['observations']['residuals'][role], selected[1]['observations']['residuals'][role]),
                                          'peak_to_next': residual_change(selected[1]['observations']['residuals'][role], selected[2]['observations']['residuals'][role])}
                                    for role in ('fitting', 'anchor')}
    result = {'summary': summary, 'cases': cases, 'contrasts': contrasts,
              'limits': ['Posthoc maxima and saved neighbors; no continuous-time or causal mechanism claim.',
                         'All supports fitted; one seed/rate and width changes optimization geometry.',
                         'Gradients/slopes are scaled Euclidean derivatives, not historical Adam directions.',
                         'Directional finite differences do not check every gradient component. Full pilot pending.']}
    lines = ['# Restored capacity spike diagnostic', '', f"Execution valid: {summary['execution_valid']}; source integrity: {summary['source_integrity']}; status: {summary['status']}.", '',
             '| Saved state | Status | Fitting raw E MSE | Mean error² | Centered variance | Offset share |', '|---|---|---:|---:|---:|---:|']
    for name, item in cases.items():
        if item['endorsed']:
            r = item['observations']['residuals']['fitting']
            lines.append(f"| {name} | verified | {r['energy_mse_eV2']:.12g} | {r['mean_energy_error_squared_eV2']:.12g} | {r['centered_energy_error_variance_eV2']:.12g} | {r['offset_share']} |")
        else: lines.append(f"| {name} | {item['summary']['status']} (unendorsed) | — | — | — | — |")
    lines += ['', 'Energy MSE and its components are in eV². Signed errors use prediction minus target. Groups, anchors, full geometry, verification and signed increment decompositions are retained in JSON.', '',
              '| Saved state | Weighted E gradient norm | Weighted F gradient norm | Backbone cosine | Head-kernel cosine | Weighted head-bias E gradient |', '|---|---:|---:|---:|---:|---:|']
    for name, item in cases.items():
        if item['endorsed']:
            g = item['observations']['gradients']['weighted']; w = g['whole']
            lines.append(f"| {name} | {w['energy_norm']:.10g} | {w['force_norm']:.10g} | {g['backbone']['cosine']} | {g['head_kernel']['cosine']} | {item['observations']['bias_check']['observed_weighted_energy']:.10g} |")
    lines += ['', *result['limits'], '', 'Zero training updates, zero random parameter initializations, zero optimizer proposals. Restored-state prediction, derivative and finite-difference calls are separately counted in per-case work.json. Raw learning gates remain unchanged.']
    write(output / 'capacity-spike-diagnostics.json', result)
    (output / 'capacity-spike-diagnostics.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/aspirin-capacity-spike.json')
    parser.add_argument('--report-only', action='store_true'); parser.add_argument('--output', default='reports')
    args = parser.parse_args(); config = read(args.config)
    if args.report_only: render(config['run_directory'], args.output)
    else:
        result = execute(config)
        print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, sort_keys=True), flush=True)
