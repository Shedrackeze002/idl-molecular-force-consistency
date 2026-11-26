"""Fresh capacity runs; public guards precede numerical imports."""
import argparse
from dataclasses import asdict
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from . import capacity_contract as cc


def load_support(config):
    source = cc.verify_sources(config)
    from . import late_rate as lr
    from .runs import code_provenance
    prov = source['provenance']
    parts = lr._parts(source['root'], {**prov, 'input_files_sha256': source['input_files_sha256']})
    return {**parts, 'norm': lr.Normalization(**prov['normalization']), 'baselines': prov['original_baselines'],
            'source_pins': source['pins'], 'source_facts': source, 'code': code_provenance(Path.cwd()),
            'source_provenance_sha256': cc.digest(Path(source['root']) / 'provenance.json'),
            'source_input_paths': {r: str(Path(source['root']) / f'{r}-inputs.npz') for r in parts}}


def verify_template(template, provenance):
    from .width_rate_runs import canonical_state_hashes
    if canonical_state_hashes(template) != provenance['initial_state_hashes']:
        raise ValueError('Independent fresh template differs from saved initialization')


def _launch_probe(root, timeout, token):
    if timeout <= 0:raise TimeoutError('No child wall budget remains')
    owner = cc.DirectoryLease(root)
    try:
        try:
            result = subprocess.run([sys.executable, '-m', 'force_consistency.capacity_runs', '--probe', str(root), '--token', token],
                                    check=True, capture_output=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            owner.write('probe-stdout.txt', exc.stdout or b'')
            owner.write('probe-stderr.txt', exc.stderr or b'')
            raise
        owner.write('probe-stdout.txt', result.stdout);owner.write('probe-stderr.txt', result.stderr)
    finally:owner.close()
    return cc.read(root / 'checkpoint-verification.json')


def _run_case(config, case, support, root, deadline, *, probe_runner=_launch_probe):
    cc.validate_case(config, case);root = Path(root).absolute()
    parent = support.get('_root_owner') or cc.DirectoryLease(root.parent)
    owns_parent = '_root_owner' not in support
    if parent.path != root.parent or root.name != case['name']:
        if owns_parent:parent.close()
        raise ValueError('Case root assignment differs')
    try:owner = parent.child(root.name, create=True)
    except BaseException:
        if owns_parent:parent.close()
        raise
    start = time.perf_counter();end = min(deadline, start + case['wall_seconds'])
    try:
        import jax
        from .runs import require_backend
        with jax.default_device(require_backend(config['backend'])):
            return _run_owned_case(config, case, support, owner, start, end, probe_runner)
    finally:
        owner.close()
        if owns_parent:parent.close()


def _run_owned_case(c, case, support, owner, start, deadline, probe_runner):
    import jax
    import numpy as np
    from . import late_rate as lr
    from .loss_balance import build_weighted_step
    root = owner.path;reserve = c['finalization_reserve_seconds']
    parts = {r: support[r] for r in ('fitting', 'anchor')};norm = support['norm'];fit_count = len(parts['fitting'].record_ids)
    computed = attempted = 0;pending = False;receipt = None;history = [];oracles = {};state = initial = None
    out = {'case': case, 'status': 'failed', 'execution_valid': False, 'updates': 0, 'attempted_updates': 0,
           'persisted_updates': 0, 'pending_update_unknown': False, 'frame_exposures': 0,
           'initialization_status': 'not_started', 'initializations': 0, 'initialization_frames': 0,
           'warmup_status': 'not_started', 'warmup_frame_exposures': 0, 'oracle_status': 'not_started',
           'scoring_status': 'not_started', 'scoring_prediction_frames': 0, 'reference_status': 'not_started',
           'reference_prediction_frames': 0, 'probe_status': 'not_started', 'child_work': None}
    def check(final=False):
        owner.check()
        if time.perf_counter() + (0 if final else reserve) >= deadline:raise TimeoutError('Case wall cap/reserve exhausted')
    def phase(name, status='started', **details):
        owner.json('phase.json', {'phase': name, 'status': status, 'exact_computed_updates': computed,
                   'attempted_updates': attempted, 'persisted_update': receipt['update'] if receipt else None,
                   'pending_update_unknown': pending, **details})
    try:
        check();cc.verify_pins(support['source_pins']);cc.verify_code(support['code'])
        owner.json('config.json', c);owner.json('case.json', case);owner.json('code.json', support['code'])
        for role, p in parts.items():
            if support.get('source_input_paths'):owner.write(f'{role}-inputs.npz', Path(support['source_input_paths'][role]).read_bytes())
            else:
                with owner.directory():np.savez(f'{role}-inputs.npz', **lr._arrays(p))
        check();phase('initialization');out['initialization_status'] = 'started'
        model = lr.EnergyModel(lr.ModelConfig(**c['model'], features=case['features']))
        batch = lr.partition_batch(parts['anchor'], np.arange(min(c['microbatch_size'], len(parts['anchor'].record_ids))))
        initial = lr.initialize_state(model, batch, norm, c['seed'], c['learning_rate'])
        jax.block_until_ready(initial);lr.validate_state(initial, norm, 0)
        out.update(initialization_status='completed', initializations=1, initialization_frames=len(batch['energy']),
                   initial_state_hashes=lr.state_hashes(initial), parameter_count=sum(np.asarray(x).size for x in jax.tree.leaves(initial['params'])))
        state = copy.deepcopy(initial);check()
        inputs = {r: {'arrays': {k: lr.array_identity(v) for k, v in lr._arrays(p).items()},
                       'record_ids': list(p.record_ids), 'memberships': p.memberships} for r, p in parts.items()}
        prov = lr.normalized({'case': case, 'config_identity': cc.canonical_hash(c), 'model': asdict(model.config),
                    'normalization': asdict(norm), 'inputs': inputs,
                    'input_files_sha256': {r: cc.digest(root / f'{r}-inputs.npz') for r in parts},
                    'original_baselines': support['baselines'], 'initial_state_hashes': out['initial_state_hashes'],
                    'source_directory': c.get('source_directory'), 'source_provenance_sha256': support['source_provenance_sha256'],
                    'source_pins': support['source_pins'], 'code': support['code'], 'precision': 'float64', 'backend': 'cpu'})
        owner.json('provenance.json', prov)
        def score():
            nonlocal receipt
            check(final=True);phase('scoring');out['scoring_status'] = 'started'
            metrics = lr.coverage_metrics(state['params'], parts['fitting'], parts['anchor'], model, c['microbatch_size'])
            lr.assert_finite(metrics);out['scoring_prediction_frames'] += sum(len(p.record_ids) for p in parts.values())
            out['scoring_status'] = 'completed'
            if int(state['update']) == 0:
                for role in parts:
                    for k in cc.MSE_KEYS:
                        if not np.isclose(metrics[role][k], support['baselines'][role][k], atol=1e-12, rtol=1e-10):
                            raise ValueError('Fresh zero-head baseline differs')
            row = {'update': int(state['update']), 'metrics': metrics, 'state_hashes': lr.state_hashes(state),
                   'rate': None if int(state['update']) == 0 else c['learning_rate']}
            for role in parts:row[role + '_gate'] = lr.gate_result(metrics[role], support['baselines'][role], c['gate_fraction'])
            history.append(row)
            with owner.directory():receipt = lr._commit(Path('.'), state, history, prov)
            owner.check();phase('scoring', 'completed');check(final=True)
            print(json.dumps({'case': case['name'], 'update': int(state['update']),
                  'energy_fraction': row['fitting_gate']['fraction_remaining'][cc.MSE_KEYS[0]],
                  'force_fraction': row['fitting_gate']['fraction_remaining'][cc.MSE_KEYS[1]],
                  'elapsed_seconds': time.perf_counter() - start}), flush=True)
        score()
        full = lr.partition_batch(parts['fitting'], np.arange(fit_count))
        batches = [lr.partition_batch(parts['fitting'], np.arange(i, min(i + c['microbatch_size'], fit_count)))
                   for i in range(0, fit_count, c['microbatch_size'])]
        check();phase('compile');step = build_weighted_step(model, norm, c['learning_rate'], c['loss_weights'])
        step.prepare(state['params'], state['optimizer'], batches);check()
        phase('warmup');out['warmup_status'] = 'started'
        warm = step(state['params'], state['optimizer'], batches);jax.block_until_ready(warm);lr.assert_finite(warm)
        out.update(warmup_status='completed', warmup_frame_exposures=fit_count)
        if lr.state_hashes(state) != out['initial_state_hashes']:raise ValueError('Warmup changed fresh state')
        phase('warmup', 'completed')
        for u in range(1, c['max_updates'] + 1):
            check();candidate = None
            if u in c['oracle_updates']:
                phase('oracle', outgoing_update=u);out['oracle_status'] = 'started'
                nonzero = any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(state['optimizer'][0].mu))
                if u > 1 and not nonzero:raise ValueError('Nonzero-moment oracle received zero moments')
                candidate, record = lr._boundary(state, c['learning_rate'], step, batches, full, model, norm)
                record['incoming_nonzero_moments'] = nonzero;oracles[str(u)] = record
                owner.json(f'oracle-{u}.json', record);out['oracle_status'] = 'completed';check()
            if u == 1 or history[-1]['update'] == u - 1:
                phase('training_block', possible_first_update=u, possible_last_update=min(x for x in c['score_updates'] if x >= u), unknown_if_unreturned=True)
            attempted += 1;pending = True
            pp, oo, metrics = step(state['params'], state['optimizer'], batches)
            jax.block_until_ready((pp, oo, metrics));lr.assert_finite((pp, oo, metrics))
            state = {**state, 'params': pp, 'optimizer': oo, 'update': u};lr.validate_state(state, norm, u)
            computed += 1;pending = False
            if candidate is not None:
                lr._close_trees((pp, oo), (candidate['params'], candidate['optimizer']))
                oracles[str(u)].update(actual_update_verified=True, committed_state_hashes=lr.state_hashes(state))
                owner.json(f'oracle-{u}.json', oracles[str(u)])
            if u in c['score_updates']:score()
        check(final=True);phase('reference');out['reference_status'] = 'started';refs = {}
        for role, p in parts.items():
            e, f = lr._partition_predictions(state['params'], p, model, c['microbatch_size']);lr.assert_finite((e, f))
            refs[role + '_energy'] = e;refs[role + '_forces'] = f;out['reference_prediction_frames'] += len(p.record_ids)
        with owner.directory():np.savez('reference.npz', **refs)
        out['reference_status'] = 'completed';check(final=True)
        owner.json('checkpoint-contract.json', {'case': case, 'commit': receipt, 'config_identity': cc.canonical_hash(c),
                   'reference_sha256': cc.digest(root / 'reference.npz'), 'provenance_sha256': cc.digest(root / 'provenance.json'),
                   'normalization': asdict(norm), 'initial_state_hashes': out['initial_state_hashes']})
        phase('probe');out['probe_status'] = 'started'
        attempt = cc.reserve_probe(root, case, c, support['code']);check(final=True)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:raise TimeoutError('No child wall budget remains')
        verified = probe_runner(root, remaining, attempt['token'])
        completion = cc.read(root / 'probe-completion.json');entry = cc.read(root / 'probe-entry.json')
        if (verified['status'] != 'passed' or verified != cc.read(root / 'checkpoint-verification.json')
            or verified['binding'] != cc.probe_binding(root) or completion['verification_sha256'] != cc.digest(root / 'checkpoint-verification.json')
            or completion['attempt_sha256'] != entry['attempt_sha256'] or entry['attempt_sha256'] != cc.digest(root / 'probe-attempt.json')):
            raise ValueError('Child receipt/completion binding differs')
        out.update(probe_status='completed', checkpoint_verification=verified, child_work=verified['work'])
        cc.verify_pins(support['source_pins']);cc.verify_code(support['code']);check(final=True)
        if lr.state_hashes(initial) != out['initial_state_hashes']:raise ValueError('Fresh initialization mutated')
        out.update(status='completed', execution_valid=True, stop_reason='fixed_budget')
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        out.update(status='incomplete' if isinstance(exc, TimeoutError) else 'failed', execution_valid=False,
                   stop_reason='wall_budget' if isinstance(exc, TimeoutError) else 'failure', failure_reason=f'{type(exc).__name__}: {exc}')
        for key in ('initialization_status', 'warmup_status', 'oracle_status', 'scoring_status', 'reference_status', 'probe_status'):
            if out[key] == 'started':out[key] = 'unknown'
        if out['initialization_status'] == 'unknown':out.update(initializations=None, initialization_frames=None)
        if isinstance(exc, subprocess.CalledProcessError):out['probe_stderr'] = str(exc.stderr)[-4000:]
    out.update(updates=computed, attempted_updates=attempted, persisted_updates=receipt['update'] if receipt else 0,
               persisted_update=receipt['update'] if receipt else None, pending_update_unknown=pending,
               frame_exposures=computed * fit_count, history=history, oracles=oracles,
               initial_state_hashes=out.get('initial_state_hashes'), final_state_hashes=lr.state_hashes(state) if state is not None else None,
               elapsed_seconds=time.perf_counter() - start, wall_seconds=case['wall_seconds'])
    if time.perf_counter() >= deadline:out.update(status='incomplete', execution_valid=False, stop_reason='wall_budget')
    if owner.owns():
        phase('case', out['status']);owner.json('summary.json', out)
        if time.perf_counter() >= deadline:
            out.update(status='incomplete', execution_valid=False, stop_reason='wall_budget', elapsed_seconds=time.perf_counter() - start)
            owner.json('summary.json', out)
    else:out.update(execution_valid=False, status='failed', failure_reason='Case directory ownership lost')
    return out


def _probe_saved(root, token, *, production=True):
    root = Path(root).absolute();entry = cc.claim_probe_entry(root, token, production=production)
    cc.audit_provenance(root, cc.read(root / 'config.json'), cc.read(root / 'case.json'))
    owner = cc.DirectoryLease(root, expected=entry['binding']['directory_identity'])
    start = time.perf_counter()
    try:
        import jax
        import numpy as np
        from . import late_rate as lr
        c = cc.read(root / 'config.json');case = cc.read(root / 'case.json');prov = cc.read(root / 'provenance.json')
        contract = cc.read(root / 'checkpoint-contract.json');receipt = cc.read(root / 'commit.json')
        cc.validate_case(c, case);cc.verify_pins(prov['source_pins']);cc.verify_code(prov['code'])
        if (prov['case'] != case or prov['model'] != {**c['model'], 'features': case['features']}
            or prov['config_identity'] != cc.canonical_hash(c) or contract['commit'] != receipt
            or contract['case'] != case or receipt['update'] != c['max_updates']
            or contract['reference_sha256'] != cc.digest(root / 'reference.npz')
            or contract['provenance_sha256'] != cc.digest(root / 'provenance.json')):
            raise ValueError('Child case/contract differs')
        cc.audit_state(root, receipt, prov)
        parts = lr._parts(root, prov);norm = lr.Normalization(**prov['normalization'])
        with jax.default_device(lr.require_backend('cpu')):
            model = lr.EnergyModel(lr.ModelConfig(**prov['model']))
            batch = lr.partition_batch(parts['anchor'], np.arange(min(c['microbatch_size'], len(parts['anchor'].record_ids))))
            owner.json('probe-phase.json', {'phase': 'template', 'status': 'started'})
            template = lr.initialize_state(model, batch, norm, c['seed'], c['learning_rate'])
            jax.block_until_ready(template);lr.validate_state(template, norm, 0);verify_template(template, prov)
            zero = lr.load_checkpoint(root / 'snapshots/0.msgpack', prov, template=template)
            if lr.state_hashes(zero) != lr.state_hashes(template):raise ValueError('Saved0 template differs')
            state = lr.load_checkpoint(root / receipt['checkpoint'], prov, template=template)
            lr.validate_state(state, norm, c['max_updates'])
            if lr.state_hashes(state) != receipt['state_hashes']:raise ValueError('Restored endpoint state differs')
            owner.json('probe-phase.json', {'phase': 'predictions', 'status': 'started', 'known_template_initializations': 1, 'known_template_frames': len(batch['energy'])})
            maximum = lr._prediction_error(state, parts, model, root / 'reference.npz', c['microbatch_size'], c['checkpoint_prediction_atol'])
        count = sum(len(p.record_ids) for p in parts.values())
        result = {'status': 'passed', 'case': case, 'binding': cc.probe_binding(root), 'update': c['max_updates'],
                  'adam_count': int(state['optimizer'][0].count), 'state_hashes': lr.state_hashes(state),
                  'initial_state_hashes': lr.state_hashes(template), 'maximum_absolute_error': maximum,
                  'precision': 'float64', 'normalization': asdict(norm), 'tolerance': c['checkpoint_prediction_atol'],
                  'distinct_prediction_structures': len(set(parts['fitting'].record_ids) | set(parts['anchor'].record_ids)),
                  'work': {'template_initializations': 1, 'template_initializer_frames': len(batch['energy']), 'prediction_frames': count, 'training_updates': 0}}
        owner.json('checkpoint-verification.json', result)
        owner.json('probe-phase.json', {'phase': 'probe', 'status': 'completed', 'work': result['work']})
        owner.json('probe-completion.json', {'status': 'passed', 'attempt_sha256': entry['attempt_sha256'],
                   'entry_sha256': cc.digest(root / 'probe-entry.json'), 'verification_sha256': cc.digest(root / 'checkpoint-verification.json'),
                   'elapsed_seconds': time.perf_counter() - start})
        owner.check();return result
    finally:owner.close()


def execute(config):
    cc.validate_contract(config)
    from .capacity_report import execute as execute_reviewed
    return execute_reviewed(config)


def main():
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config');group.add_argument('--probe');parser.add_argument('--token')
    args = parser.parse_args()
    if args.probe:
        if not args.token:parser.error('A reserved probe token is required')
        _probe_saved(args.probe, args.token, production=True);return 0
    return 0 if execute(cc.read(args.config))['execution_valid'] else 1


if __name__ == '__main__':raise SystemExit(main())
