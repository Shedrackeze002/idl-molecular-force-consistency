"""Capacity orchestration and saved-only reconciliation; no model work in reports."""
import hashlib
import json
import math
from pathlib import Path
import time

from . import capacity_contract as cc
from .width_rate_replacement import json_bytes, journal

REPORT_JSON = Path('reports/capacity-diagnostics.json')
REPORT_MD = Path('reports/capacity-diagnostics.md')
ROLES = ('fitting', 'anchor')
LIMITS = [
    'One seed, one selected learning rate, fitted supports and a finite update budget; full pilot pending.',
    'Same seed declares the initialization rule across different shapes, not identical parameters.',
    'Width changes parameterization and optimization geometry together; no isolated expressivity mechanism is established.',
    'Sparse scores do not prove continuous stability, convergence or identify unsampled transients.',
    'Raw energy and force gates retain original update0 baselines; residual decomposition never recenters acceptance.',
    'All anchors are fitted; 144 production role predictions represent 128 distinct structures, not held-out validation.',
    'Sequential CPU wall times are engineering records, not hardware efficiency rankings or GPU parity evidence.',
    'This capacity diagnostic is not a result for the main force-consistency hypothesis.'
]


def metric_contrast(narrower, wider):
    return {'wider_minus_narrower': wider - narrower,
            'wider_divided_by_narrower': wider / narrower if narrower else None,
            'ratio_unavailable_reason': None if narrower else 'narrower denominator is zero'}


def evidence(c, identity, title, value):
    # Append only, including operational failures and invalid scientific executions.
    with cc.no_symlinks(c['evidence_log']).open('a') as f:
        f.write(f'\n## {identity} — {title}\n\n```json\n' + json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n```\n')
        f.flush()


def inventory(root):
    root = cc.no_symlinks(root)
    return {str(p.relative_to(root)): cc.digest(cc.no_symlinks(p)) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix() not in ('raw-artifacts.json', 'finalization.json')}


def freeze_inventory(root, owner=None):
    value = inventory(root)
    if owner:owner.json('raw-artifacts.json', value)
    else:cc.atomic(Path(root) / 'raw-artifacts.json', value)
    return value


def _child_work(root, c, case, receipt, prov):
    child = cc.read(root / 'checkpoint-verification.json');attempt = cc.read(root / 'probe-attempt.json')
    entry = cc.read(root / 'probe-entry.json');completion = cc.read(root / 'probe-completion.json')
    binding = cc.probe_binding(root)
    count = sum(len(prov['inputs'][r]['record_ids']) for r in ROLES)
    distinct = len(set(prov['inputs']['fitting']['record_ids']) | set(prov['inputs']['anchor']['record_ids']))
    work = {'template_initializations': 1, 'template_initializer_frames': min(c['microbatch_size'], len(prov['inputs']['anchor']['record_ids'])),
            'prediction_frames': count, 'training_updates': 0}
    maximum = child['maximum_absolute_error']
    if (attempt['ordinal'] != 1 or len(attempt['token']) != 64 or any(x not in '0123456789abcdef' for x in attempt['token'])
        or attempt['binding'] != binding or entry['binding'] != binding or child['binding'] != binding
        or entry['attempt_sha256'] != cc.digest(root / 'probe-attempt.json')
        or completion['attempt_sha256'] != entry['attempt_sha256'] or completion['status'] != 'passed'
        or completion['entry_sha256'] != cc.digest(root / 'probe-entry.json')
        or completion['verification_sha256'] != cc.digest(root / 'checkpoint-verification.json')
        or child['status'] != 'passed' or child['case'] != case or child['update'] != c['max_updates']
        or child['adam_count'] != c['max_updates'] or child['state_hashes'] != receipt['state_hashes']
        or child['initial_state_hashes'] != prov['initial_state_hashes'] or child['normalization'] != prov['normalization']
        or child['precision'] != 'float64' or child['tolerance'] != c['checkpoint_prediction_atol']
        or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or not 0 <= maximum <= c['checkpoint_prediction_atol']
        or child['work'] != work or child['distinct_prediction_structures'] != distinct
        or not math.isfinite(completion['elapsed_seconds']) or completion['elapsed_seconds'] < 0):
        raise ValueError('Child attempt/entry/completion or work binding differs')
    return child, work, distinct


def _validated_case(root, c, case, *, source=None):
    import numpy as np
    from . import late_rate as lr
    root = cc.no_symlinks(root);prov = cc.audit_provenance(root, c, case, source=source)
    cc.verify_pins(prov['source_pins']);cc.verify_code(prov['code'])
    s = cc.read(root / 'summary.json');receipt = cc.read(root / 'commit.json');contract = cc.read(root / 'checkpoint-contract.json')
    history = cc.read(root / receipt['history']);end = c['max_updates'];n = len(prov['inputs']['fitting']['record_ids'])
    pred_count = n + len(prov['inputs']['anchor']['record_ids'])
    if (not s['execution_valid'] or s['status'] != 'completed' or s['case'] != case or s['stop_reason'] != 'fixed_budget'
        or receipt != contract['commit'] or receipt['update'] != end or receipt['history'] != f'histories/{end}.json'
        or cc.digest(root / receipt['history']) != receipt['history_sha256'] or contract['case'] != case
        or contract['config_identity'] != cc.canonical_hash(c) or contract['normalization'] != prov['normalization']
        or contract['provenance_sha256'] != cc.digest(root / 'provenance.json')
        or contract['reference_sha256'] != cc.digest(root / 'reference.npz')
        or contract['initial_state_hashes'] != prov['initial_state_hashes']
        or [r['update'] for r in history] != c['score_updates'] or s['history'] != history
        or s['updates'] != end or s['attempted_updates'] != end or s['persisted_updates'] != end or s['persisted_update'] != end
        or s['pending_update_unknown'] or s['frame_exposures'] != end * n
        or not math.isfinite(s['elapsed_seconds']) or not 0 <= s['elapsed_seconds'] < case['wall_seconds']
        or s['wall_seconds'] != case['wall_seconds']):
        raise ValueError('Saved case completion, score sequence, counters or cap differ')
    last = cc.audit_state(root, receipt, prov)
    if (s['initial_state_hashes'] != prov['initial_state_hashes'] or s['final_state_hashes'] != last['state_hashes']
        or s['parameter_count'] != last['parameter_count']):raise ValueError('Width/state hash/parameter count differs')
    for i, row in enumerate(history):
        u = row['update'];cp = cc.no_symlinks(root / f'snapshots/{u}.msgpack')
        snapshot = cc.audit_state(root, {'update': u, 'checkpoint': f'snapshots/{u}.msgpack',
                                  'checkpoint_sha256': cc.digest(cp), 'state_hashes': row['state_hashes']}, prov)
        if snapshot['parameter_count'] != last['parameter_count']:raise ValueError('Width snapshot shape differs')
        if cc.read(root / f'histories/{u}.json') != history[:i + 1]:raise ValueError('Immutable history prefix differs')
        if row['rate'] != (None if u == 0 else c['learning_rate']):raise ValueError('Saved rate differs')
        lr.assert_finite(row['metrics'])
        for role in ROLES:
            if any(row['metrics'][role][k] < 0 for k in cc.MSE_KEYS):raise ValueError('Negative raw MSE')
            if row[role + '_gate'] != lr.gate_result(row['metrics'][role], prov['original_baselines'][role], c['gate_fraction']):
                raise ValueError('Original-baseline raw gate differs')
    for role in ROLES:
        for key in cc.MSE_KEYS:
            if not np.isclose(history[0]['metrics'][role][key], prov['original_baselines'][role][key], atol=1e-12, rtol=1e-10):
                raise ValueError('Fresh zero-head original baseline differs')
    child, child_work, distinct = _child_work(root, c, case, receipt, prov)
    if s['checkpoint_verification'] != child or s['child_work'] != child_work:raise ValueError('Summary child work differs')
    expected = {'initialization_status': 'completed', 'initializations': 1,
                'initialization_frames': min(c['microbatch_size'], len(prov['inputs']['anchor']['record_ids'])),
                'warmup_status': 'completed', 'warmup_frame_exposures': n, 'scoring_status': 'completed',
                'scoring_prediction_frames': len(history) * pred_count, 'reference_status': 'completed',
                'reference_prediction_frames': pred_count, 'oracle_status': 'completed', 'probe_status': 'completed'}
    if any(s[k] != v for k, v in expected.items()):raise ValueError('Initialization/warmup/prediction work differs')
    oracles = {}
    for u in c['oracle_updates']:
        o = cc.read(root / f'oracle-{u}.json')
        before = next(r for r in history if r['update'] == u - 1);after = next(r for r in history if r['update'] == u)
        if (o['status'] != 'passed' or not o['actual_update_verified'] or not o['discarded']
            or o['incoming_update'] != u - 1 or o['outgoing_update'] != u or o['rate'] != c['learning_rate']
            or o['incoming_nonzero_moments'] != (u > 1) or o['incoming_state_hashes'] != before['state_hashes']
            or o['proposal_state_hashes'] != after['state_hashes'] or o['committed_state_hashes'] != after['state_hashes']
            or o['atol'] != 1e-10 or o['rtol'] != 1e-8
            or not math.isfinite(o['maximum_absolute_difference']) or o['maximum_absolute_difference'] < 0
            or o['fullbatch_gradient_frames'] != n or o['accumulated_gradient_frames'] != n
            or o['independent_numpy_proposals'] != 1 or o['accumulated_proposals'] != 1
            or s['oracles'].get(str(u)) != o):raise ValueError('Oracle incoming/actual committed state or discarded work differs')
        oracles[str(u)] = o
    if set(s['oracles']) != {str(u) for u in c['oracle_updates']}:raise ValueError('Oracle sequence differs')
    parts = lr._parts(root, prov);residuals = {}
    with np.load(root / 'reference.npz', allow_pickle=False) as ref:
        if set(ref.files) != {r + suffix for r in ROLES for suffix in ('_energy', '_forces')}:raise ValueError('Reference roles differ')
        for role, p in parts.items():
            e, f = ref[role + '_energy'], ref[role + '_forces']
            if e.shape != p.energy_residual.shape or f.shape != p.forces.shape:raise ValueError('Reference shapes differ')
            residuals[role] = lr.residual_summary(e, p.energy_residual, f, p.forces)
            rebuilt = lr._group_score(p, e, f);saved = history[-1]['metrics'][role]
            for key in cc.MSE_KEYS:
                if not np.isclose(rebuilt[key], saved[key], atol=1e-12, rtol=1e-10):raise ValueError('Endpoint raw metric differs from saved predictions')
            if lr.normalized(rebuilt['per_group']) != saved['per_group'] or lr.normalized(rebuilt['equal_group']) != saved['equal_group']:
                raise ValueError('Endpoint group metrics differ from saved predictions')
    peaks = {r: {k: {'update': max(history, key=lambda x: x['metrics'][r][k])['update'],
                      'raw_mse': max(x['metrics'][r][k] for x in history),
                      'original_baseline_fraction': max(x['metrics'][r][k] for x in history) / prov['original_baselines'][r][k]}
                    for k in cc.MSE_KEYS} for r in ROLES}
    # At production this is 5500 -> 6000; tiny fixtures use their penultimate score.
    terminal_start = end - 500 if end >= 500 else c['score_updates'][-2]
    old = next(x for x in history if x['update'] == terminal_start)
    terminal = {r: {k: {'relative_change': (history[-1]['metrics'][r][k] - old['metrics'][r][k]) / old['metrics'][r][k]
                            if old['metrics'][r][k] else None,
                         'unavailable_reason': None if old['metrics'][r][k] else 'terminal-window baseline is zero'} for k in cc.MSE_KEYS} for r in ROLES}
    return {'status': 'completed', 'execution_valid': True, 'case': case, 'updates': end, 'attempted_updates': end,
            'persisted_updates': end, 'verified_persisted_updates': end, 'frame_exposures': end * n, 'pending_update_unknown': False,
            'history': history, 'endpoint_metrics': history[-1]['metrics'],
            'original_baseline_fractions': {r: history[-1][r + '_gate']['fraction_remaining'] for r in ROLES},
            'learning_accepted': all(history[-1][r + '_gate']['accepted'] for r in ROLES),
            'residual_decomposition': residuals, 'sampled_peaks': peaks,
            'terminal_window': {'from_update': terminal_start, 'to_update': end, 'metrics': terminal},
            'initial_state_hashes': prov['initial_state_hashes'], 'parameter_count': last['parameter_count'],
            'child_maximum_prediction_error': child['maximum_absolute_error'], 'distinct_prediction_structures': distinct,
            'oracles': oracles, 'work': {**expected, 'oracle_fullbatch_gradient_frames': len(oracles) * n,
                'oracle_accumulated_gradient_frames': len(oracles) * n, 'independent_numpy_proposals': len(oracles),
                'discarded_accumulated_proposals': len(oracles)}, 'child_work': child_work, 'elapsed_seconds': s['elapsed_seconds']}


def _partial_case(path, c, summary):
    path = Path(path);summary = summary or {};out = {'status': summary.get('status', 'partial_or_blocked'), 'execution_valid': False,
             'raw_summary': summary or None, 'child_work': None, 'verified_persisted_updates': None,
             'pending_update_unknown': bool(not summary or summary.get('pending_update_unknown') or summary.get('work_since_last_receipt_unknown'))}
    for k in ('updates', 'attempted_updates', 'persisted_updates', 'frame_exposures'):
        v = summary.get(k);out[k] = v if type(v) is int and v >= 0 else None
    try:
        prov = cc.read(path / 'provenance.json');receipt = cc.read(path / 'commit.json');history = cc.read(path / receipt['history'])
        if (cc.digest(path / receipt['history']) != receipt['history_sha256'] or history[-1]['update'] != receipt['update']
            or not 0 <= receipt['update'] <= c['max_updates'] or history[-1]['state_hashes'] != receipt['state_hashes']):raise ValueError('Partial receipt differs')
        cc.audit_state(path, receipt, prov);out['verified_persisted_updates'] = receipt['update']
        try:out['child_work'] = _child_work(path, c, cc.read(path / 'case.json'), receipt, prov)[1]
        except Exception:pass
    except Exception:
        if summary.get('status') == 'blocked' and summary.get('updates') == 0:out['verified_persisted_updates'] = 0
    out['work'] = {k: v for k, v in summary.items() if any(k.startswith(prefix) for prefix in ('initializ', 'warmup', 'oracle', 'scoring', 'reference', 'probe'))}
    out['reported_child_work'] = summary.get('child_work')
    out['child_phase'] = cc.read(path / 'probe-phase.json') if (path / 'probe-phase.json').is_file() else None
    phase = out['child_phase'] or {}
    out['partial_child_work'] = {'reported_template_initializations': phase.get('known_template_initializations'),
                                'reported_template_initializer_frames': phase.get('known_template_frames'),
                                'prediction_frames': None,
                                'scope': 'Child phase records returned initializer work; absent completion does not establish prediction count.'}
    out['last_phase'] = cc.read(path / 'phase.json') if (path / 'phase.json').is_file() else None
    out['work_accounting_status'] = 'Returned reported computation; independently receipt-verified persisted counts where available; unknown work is not zero.'
    return out


def _closure_errors(root, c):
    closed = cc.read(root / 'finalization.json');prefix = cc.no_symlinks(c['evidence_log']).read_bytes()[:closed['evidence_prefix_bytes']]
    if (not math.isfinite(closed['elapsed_seconds']) or closed['elapsed_seconds'] < 0
        or closed['raw_inventory_sha256'] != cc.digest(root / 'raw-artifacts.json')
        or closed['report_sha256'] != cc.digest(REPORT_JSON) or closed['report_md_sha256'] != cc.digest(REPORT_MD)
        or hashlib.sha256(prefix).hexdigest() != closed['evidence_prefix_sha256']):raise ValueError('Final report/evidence/marker cap closure invalid')
    if not closed['within_global_cap'] or closed['elapsed_seconds'] >= c['total_wall_seconds']:
        raise TimeoutError('Global cap exceeded during finalization')


def build_report(root, *, allow_pending_finalization=False):
    c = cc.read(Path(root) / 'config.json');cc.validate_contract(c)
    return _build_report(root, production=True, allow_pending_finalization=allow_pending_finalization)


def _build_report(root, *, production, allow_pending_finalization=False):
    root = cc.no_symlinks(root);c = cc.read(root / 'config.json');errors = [];cases = {};source = None;returned = {}
    try:
        returned = cc.read(root / 'cases.json')
        if (set(returned) != {x['name'] for x in c['cases']}
            or any(returned[x['name']]['case'] != x for x in c['cases'])):raise ValueError('Controller case assignments differ')
    except Exception as exc:errors.append(f'Controller records: {exc}')
    try:
        if production:cc.validate_contract(c);source = cc.verify_sources(c)
        cc.verify_code(cc.read(root / 'code.json'))
        saved_inventory = cc.read(root / 'raw-artifacts.json')
        if inventory(root) != saved_inventory:raise ValueError('Raw inventory or artifact set differs')
        cc.verify_pins({str(root / name): sha for name, sha in saved_inventory.items()})
        completion = cc.read(root / 'completion.json')
        if (not completion['execution_valid'] or not math.isfinite(completion['elapsed_seconds'])
            or not 0 <= completion['elapsed_seconds'] < c['total_wall_seconds']
            or completion['case_names'] != [x['name'] for x in c['cases']]):raise ValueError('Controller completion/global cap differs')
    except (Exception, KeyboardInterrupt) as exc:errors.append(f'Global verification: {type(exc).__name__}: {exc}')
    for case in c['cases']:
        path = root / case['name']
        try:
            a = _validated_case(path, c, case, source=source)
            if cc.read(path / 'code.json') != cc.read(root / 'code.json'):raise ValueError('Case code differs from launch')
            if returned.get(case['name']) != cc.read(path / 'summary.json'):raise ValueError('Controller/local summary differs')
            cases[case['name']] = a
        except (Exception, KeyboardInterrupt) as exc:
            errors.append(f"{case['name']}: {type(exc).__name__}: {exc}")
            local = cc.read(path / 'summary.json') if (path / 'summary.json').is_file() else None
            raw = returned.get(case['name']) or local
            cases[case['name']] = {**_partial_case(path, c, raw), 'local_raw_summary': local,
                                  'reported_work_source': 'controller-returned case record' if returned.get(case['name']) else 'local summary or unknown',
                                  'verification_error': str(exc)}
    if not errors:
        a, b = cases.values()
        if a['history'][0]['metrics'] != b['history'][0]['metrics']:
            errors.append('Fresh widths do not share identical original raw baselines')
        if production and (a['parameter_count'] != 13569 or b['parameter_count'] <= a['parameter_count']):
            errors.append('Production width-specific parameter count differs')
    if not allow_pending_finalization:
        try:_closure_errors(root, c)
        except TimeoutError as exc:errors.append(str(exc))
        except (Exception, KeyboardInterrupt) as exc:errors.append(f'Finalization: {exc}')
    valid = not errors and all(a['execution_valid'] for a in cases.values())
    contrast = None
    if valid:
        a, b = cases.values()
        contrast = {r: {k: metric_contrast(a['endpoint_metrics'][r][k], b['endpoint_metrics'][r][k]) for k in cc.MSE_KEYS} for r in ROLES}
    return {'status': 'completed' if valid else 'partial_or_invalid', 'execution_valid': valid, 'contrasts_available': valid,
            'cases': cases, 'endpoint_contrasts': contrast,
            'contrast_orientation': 'width64 minus width32 and width64 divided by width32 in production',
            'updates': sum(a.get('updates') or 0 for a in cases.values()),
            'frame_exposures': sum(a.get('frame_exposures') or 0 for a in cases.values()),
            'reported_persisted_updates': sum(a.get('persisted_updates') or 0 for a in cases.values()),
            'verified_persisted_updates': sum(a.get('verified_persisted_updates') or 0 for a in cases.values()),
            'any_pending_work_unknown': any(a.get('pending_update_unknown') or a.get('updates') is None for a in cases.values()),
            'work_totals_scope': 'Known returned counts; zero aggregate contributions from unknown cases are not claims of zero work.',
            'any_learning_accepted': valid and any(a['learning_accepted'] for a in cases.values()),
            'verification_errors': errors, 'limits': LIMITS, 'full_pilot': 'pending',
            'raw_inventory_sha256': cc.digest(root / 'raw-artifacts.json') if (root / 'raw-artifacts.json').is_file() else None}


def markdown(report):
    lines = ['# Controlled capacity diagnostic', '', f"Execution valid: {report['execution_valid']}; full pilot: pending.", '',
             '| Width | Parameters | Fitting E fraction | Fitting F fraction | Both-role joint gate |',
             '|---|---:|---:|---:|---|']
    for name, a in report['cases'].items():
        if a['execution_valid']:
            f = a['original_baseline_fractions']['fitting']
            lines.append(f"| {name} | {a['parameter_count']} | {f[cc.MSE_KEYS[0]]:.10f} | {f[cc.MSE_KEYS[1]]:.10f} | {a['learning_accepted']} |")
        else:lines.append(f'| {name} | unavailable | unavailable | unavailable | unavailable |')
    lines += ['', f"Known returned updates: {report['updates']}; fitting exposures: {report['frame_exposures']}; receipt-verified persisted updates: {report['verified_persisted_updates']}.", '',
              'Contrast orientation: width64 minus width32; ratio width64 divided by width32. JSON retains anchors, groups, equal-group scores, residual decomposition, full sampled peaks and all work categories.', '',
              *['- ' + x for x in report['limits']]]
    if report['verification_errors']:lines += ['', 'Verification errors:', *['- ' + x for x in report['verification_errors']]]
    return '\n'.join(lines) + '\n'


def write_report(report):
    cc.atomic(REPORT_JSON, report)
    # The same byte serializer is used for deterministic saved-only regeneration.
    from .runs import atomic_write
    atomic_write(REPORT_MD, markdown(report).encode())


def verify_saved_report(report):
    if REPORT_JSON.read_bytes() != json_bytes(report) or REPORT_MD.read_bytes() != markdown(report).encode():
        raise ValueError('Saved JSON/Markdown differs from canonical regenerated report')
    return True


def finalize(c, owner, start, report):
    evidence(c, 'CAPACITY-FINALIZATION-001', 'post-report closure starting',
             {'report_sha256_before_closure': cc.digest(REPORT_JSON), 'status': 'finalizing; immutable marker decides cap acceptance'})
    def invalidate():
        report.update(status='partial_or_invalid', execution_valid=False, contrasts_available=False, endpoint_contrasts=None, any_learning_accepted=False)
        if 'Global cap exceeded during finalization' not in report['verification_errors']:
            report['verification_errors'].append('Global cap exceeded during finalization')
        write_report(report)
    elapsed = time.perf_counter() - start
    if elapsed >= c['total_wall_seconds']:invalidate()
    prefix = cc.no_symlinks(c['evidence_log']).read_bytes()
    closure = {'within_global_cap': elapsed < c['total_wall_seconds'], 'elapsed_seconds': time.perf_counter() - start,
               'raw_inventory_sha256': cc.digest(owner.path / 'raw-artifacts.json'), 'report_sha256': cc.digest(REPORT_JSON),
               'report_md_sha256': cc.digest(REPORT_MD), 'evidence_prefix_bytes': len(prefix),
               'evidence_prefix_sha256': hashlib.sha256(prefix).hexdigest(),
               'scope': 'Numerical work plus primary reports and evidence through marker; saved-only figures/archive/reviews/Obsidian separately timed.'}
    owner.json('finalization.json', closure)
    after = time.perf_counter() - start
    if after >= c['total_wall_seconds']:
        invalidate();closure.update(within_global_cap=False, elapsed_seconds=after,
                                   report_sha256=cc.digest(REPORT_JSON), report_md_sha256=cc.digest(REPORT_MD))
        owner.json('finalization.json', closure)
    return report


def execute(c):
    cc.validate_contract(c)
    repository = Path.cwd()
    root = cc.no_symlinks(c['run_directory']);root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False);owner = cc.DirectoryLease(root);start = time.perf_counter();deadline = start + c['total_wall_seconds']
    results = {};failure = None;current = None;source = None
    try:
        owner.json('config.json', c)
        from .runs import code_provenance
        from . import capacity_runs as runs
        code = code_provenance(Path.cwd());owner.json('code.json', code)
        snapshot = owner.child('code-snapshot', create=True)
        try:
            with snapshot.directory():
                # Anchored destination; source paths are absolute because CWD is temporarily the snapshot.
                for name in code['code_files']:
                    source_file = repository / name
                    destination = Path(name);destination.parent.mkdir(parents=True, exist_ok=True);destination.write_bytes(source_file.read_bytes())
            snapshot.check()
        finally:snapshot.close()
        owner.json('phase.json', {'phase': 'source_preflight', 'status': 'started', 'unknown_if_unreturned': True})
        evidence(c, 'CAPACITY-PRE-001', 'once-only fresh-width production preregistration',
                 {'config': c, 'config_identity': cc.canonical_hash(c), 'code': code, 'limits': LIMITS, 'status': 'starting'})
        support = runs.load_support(c);support['code'] = code;support['_root_owner'] = owner
        source = support['source_facts'];owner.json('source-facts.json', source)
        for case in c['cases']:
            current = case
            if failure:
                child = owner.child(case['name'], create=True)
                result = {'case': case, 'status': 'blocked', 'execution_valid': False, 'failure_reason': failure,
                          'updates': 0, 'attempted_updates': 0, 'persisted_updates': 0, 'frame_exposures': 0, 'pending_update_unknown': False,
                          'child_work': None}
                try:child.json('summary.json', result)
                finally:child.close()
            else:
                if time.perf_counter() + c['finalization_reserve_seconds'] >= deadline:raise TimeoutError('Global reserve exhausted before case')
                owner.json('phase.json', {'phase': 'case', 'case': case, 'status': 'started', 'unknown_if_unreturned': True})
                result = runs._run_case(c, case, support, root / case['name'], deadline)
                if not result['execution_valid']:failure = f"{case['name']}: {result.get('failure_reason', result.get('stop_reason', 'invalid execution'))}"
            results[case['name']] = result;owner.json('cases.json', results)
            journal(root, 'case', result['status'], {'case': case, 'result': result}, owner=owner)
            evidence(c, 'CAPACITY-' + case['name'].upper() + '-OUTCOME-001', 'immediate actual case outcome',
                     {'root': str(root / case['name']), 'summary_sha256': cc.digest(root / case['name'] / 'summary.json')
                         if (root / case['name'] / 'summary.json').is_file() else None, 'result': result, 'limits': LIMITS})
        cc.verify_sources(c);cc.verify_code(code)
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        failure = f'{type(exc).__name__}: {exc}'
        if not owner.owns():raise
        owner.json('controller-failure.json', {'failure_reason': failure, 'returned_cases': list(results),
                   'unreturned_work_unknown': current is not None and current['name'] not in results})
        for case in c['cases']:
            if case['name'] in results:continue
            path = root / case['name'];exists = path.is_dir()
            child = owner.child(case['name'], create=not exists)
            try:
                if (path / 'summary.json').is_file():result = cc.read(path / 'summary.json')
                else:
                    started = exists or current is not None and current['name'] == case['name']
                    result = {'case': case, 'status': 'failed' if started else 'blocked', 'execution_valid': False,
                              'failure_reason': failure, 'work_since_last_receipt_unknown': started, 'pending_update_unknown': started,
                              'updates': None if started else 0, 'attempted_updates': None if started else 0,
                              'persisted_updates': None if started else 0, 'frame_exposures': None if started else 0, 'child_work': None}
                    child.json('summary.json', result)
                results[case['name']] = result
            finally:child.close()
        owner.json('cases.json', results)
        evidence(c, 'CAPACITY-CONTROLLER-FAILURE-001', 'actual controller failure; later cases blocked',
                 {'failure_reason': failure, 'results': results, 'limits': LIMITS})
    elapsed = time.perf_counter() - start
    completion = {'execution_valid': failure is None and elapsed < c['total_wall_seconds'], 'elapsed_seconds': elapsed,
                  'failure_reason': failure, 'case_names': [x['name'] for x in c['cases']], 'full_pilot': 'pending'}
    try:
        owner.json('completion.json', completion);owner.json('phase.json', {'phase': 'matrix', 'status': 'completed' if completion['execution_valid'] else 'failed'})
        freeze_inventory(root, owner)
        report = build_report(root, allow_pending_finalization=True);write_report(report)
        evidence(c, 'CAPACITY-REPORT-001', 'saved-only reconciled actual report',
                 {'report_sha256': cc.digest(REPORT_JSON), 'report_md_sha256': cc.digest(REPORT_MD),
                  'execution_valid': report['execution_valid'], 'verification_errors': report['verification_errors'],
                  'updates': report['updates'], 'frame_exposures': report['frame_exposures'], 'endpoint_contrasts': report['endpoint_contrasts'],
                  'cases': report['cases'], 'limits': LIMITS})
        return finalize(c, owner, start, report)
    finally:owner.close()
