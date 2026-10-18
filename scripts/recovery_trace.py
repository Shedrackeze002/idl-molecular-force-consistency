"""Observe the frozen relaxer without changing its inputs or force objects."""
import io
import json
import os
from pathlib import Path
import time

import numpy as np


class TraceBudget:
    """One shared invocation/deadline/storage budget across all relaxations."""
    def __init__(self, max_queries, deadline, max_bytes):
        self.max_queries = max_queries
        self.deadline = deadline
        self.max_bytes = max_bytes
        self.invocations = 0
        self.relax_attempts = 0
        self.bytes_written = 0

    def admit(self):
        if self.invocations >= self.max_queries:
            raise RuntimeError('Trace invocation budget exhausted')
        if time.perf_counter() >= self.deadline:
            raise RuntimeError('Trace numerical deadline reached')
        # Includes the next query, per-call summary and terminal evidence.
        if self.bytes_written + 65536 + 2**23 > self.max_bytes:
            raise RuntimeError('Trace storage reservation unavailable')

    def write(self, stream, data):
        if self.bytes_written + len(data) > self.max_bytes:
            raise RuntimeError('Trace storage cap reached')
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        self.bytes_written += len(data)


class TraceSupplier:
    def __init__(self, force_fn, output_dir, max_queries, deadline, max_bytes, *, budget=None):
        self.force_fn = force_fn
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_queries = max_queries
        self.budget = budget or TraceBudget(max_queries, deadline, max_bytes)
        self.queries = []
        self.admission_failures = []

    def _event(self, event):
        with (self.output/'queries.jsonl').open('ab') as stream:
            self.budget.write(stream, (json.dumps(event, sort_keys=True, allow_nan=False)+'\n').encode())

    def _array(self, array):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        with (self.output/'arrays.bin').open('ab') as stream:
            offset = stream.tell()
            self.budget.write(stream, buffer.getvalue())
        return offset

    def __call__(self, positions):
        self.budget.relax_attempts += 1
        try:
            if len(self.queries) >= self.max_queries:
                raise RuntimeError('Per-relaxation trace budget exhausted')
            self.budget.admit()
        except RuntimeError as exc:
            event={'event':'admission_refused','relax_attempt':self.budget.relax_attempts,
                   'supplier_invoked':False,'reason':str(exc)}
            self.admission_failures.append(event)
            self._event(event)
            raise
        q = {'index':len(self.queries), 'positions':positions.copy(),
             'forces':None, 'state':'attempted'}
        offset = self._array(q['positions'])
        self._event({'event':'attempted', 'index':q['index'], 'positions_offset':offset})
        self.queries.append(q)
        self.budget.invocations += 1
        try:
            raw = self.force_fn(positions)
        except BaseException as exc:
            q['state'] = 'raised'
            q['exception_type'] = type(exc).__name__
            # Avoid turning an exception's custom str/repr into an observation.
            self._event({'event':'raised', 'index':q['index'], 'exception_type':type(exc).__name__})
            raise
        q['state'] = 'returned'
        event = {'event':'returned', 'index':q['index'], 'return_type':type(raw).__name__}
        # The production supplier returns a numeric ndarray. Never coerce an
        # arbitrary return a second time: the original relaxer owns conversion.
        if isinstance(raw, np.ndarray) and raw.dtype.kind in 'biuf':
            q['forces'] = raw.copy()
            event['forces_offset'] = self._array(q['forces'])
        self._event(event)
        return raw


def reconstruct_trace(queries, relaxation_result, procedure):
    """Check each observed decision against the unchanged procedure."""
    if len(queries) != relaxation_result['force_evaluation_count'] or not queries:
        raise ValueError('Query count differs from relax result')
    tol = procedure['convergence_tolerance_eV_per_A']
    floor = procedure['min_backtracking_floor']
    factor = procedure['backtracking_factor']
    cap = procedure['max_force_evaluations']
    max_trials = procedure['max_backtracking_attempts_per_step']
    scale = procedure['initial_step_size']
    steps = relaxation_result['steps']
    out = []
    accepted_steps = []
    parent = 0
    trials = 0
    failure = None
    converged = False
    maxima = []
    for i, q in enumerate(queries):
        row = dict(q, role='start' if i == 0 else 'trial', parent_query=None if i == 0 else parent,
                   scale=None if i == 0 else scale, accepted=False, agreement=None)
        if q['index'] != i:
            raise ValueError('Query indices differ')
        if i:
            if converged or failure or trials >= max_trials:
                raise ValueError('Query after terminal decision')
            previous = queries[parent]
            expected = previous['positions'] + scale * previous['forces']
            if not np.array_equal(expected, q['positions']):
                raise ValueError('Observed candidate differs from frozen update')
            trials += 1
        f = q['forces']
        valid = (q['state'] == 'returned' and f is not None and
                 f.shape == q['positions'].shape and np.isfinite(f).all())
        if not valid:
            if i != len(queries)-1:
                raise ValueError('Force refusal is not terminal')
            failure = relaxation_result['failure_reason']
            if failure not in ('invalid_force_shape','nonfinite_force'):
                raise ValueError('Observed refusal differs from result')
            row['decision'] = 'refused'
            out.append(row)
            break
        maximum = float(np.max(np.abs(f)))
        maxima.append(maximum)
        if i:
            row['agreement'] = float(np.sum(f * queries[parent]['forces']))
        if maximum <= tol:
            converged = True
            row['accepted'] = bool(i)
            row['decision'] = 'converged_on_trial' if i else 'converged_at_start'
            if i:
                accepted_steps.append({'accepted_step_size':scale,'trial_count':trials})
                parent = i
        elif not i:
            row['decision'] = 'start'
        else:
            agreement = row['agreement']
            if agreement >= -1e-12:
                row['accepted'] = True
                row['decision'] = 'accepted'
                accepted_steps.append({'accepted_step_size':scale,'trial_count':trials})
                parent = i
                trials = 0
                scale = min(scale*2., 1.)
            else:
                row['decision'] = 'rejected'
                scale = max(scale*factor, floor)
        out.append(row)
    if not failure and not converged:
        known = queries[parent]['forces']
        if out[-1]['decision'] == 'rejected':
            if trials >= max_trials:
                failure = 'no_progress'
            elif len(queries) >= cap:
                failure = 'budget_exhausted'
            else:
                raise ValueError('Incomplete rejected search tail')
        elif float(np.max(np.abs(scale*known))) <= 1e-9:
            failure = 'no_progress'
        elif trials >= max_trials:
            failure = 'no_progress'
        elif len(queries) >= cap:
            failure = 'budget_exhausted'
        else:
            raise ValueError('Incomplete trace without declared terminal decision')
    if accepted_steps != steps or len(steps) != relaxation_result['step_count']:
        raise ValueError('Observed accepted steps differ')
    if failure != relaxation_result['failure_reason'] or converged != relaxation_result['converged']:
        raise ValueError('Observed stop decision differs')
    if maxima != relaxation_result['force_history']['force_maximums']:
        raise ValueError('Observed force history differs')
    final = np.asarray(relaxation_result['final_positions'])
    last = queries[parent]['positions']
    if final.ndim == 2:
        last = last[0]
    if not np.array_equal(final, last):
        raise ValueError('Observed final accepted position differs')
    return {'status':'verified','queries':out,'relaxation':relaxation_result}


def graph_summary(positions, forces, cutoff_A):
    p=np.asarray(positions).reshape(-1,3);f=np.asarray(forces).reshape(-1,3)
    n=len(p);distance=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    edges=[(i,j) for i in range(n) for j in range(i+1,n) if distance[i,j]<cutoff_A]
    neighbors=[set() for _ in range(n)]
    for i,j in edges:
        neighbors[i].add(j);neighbors[j].add(i)
    unseen=set(range(n));components=[]
    while unseen:
        todo=[min(unseen)];member=set()
        while todo:
            i=todo.pop()
            if i in member:continue
            member.add(i);todo.extend(neighbors[i]-member)
        unseen-=member;components.append(sorted(member))
    pairs=distance[np.triu_indices(n,1)]
    return {'interaction_component_count':len(components),'interaction_pair_count':len(edges),
            'interaction_edges':[list(e) for e in edges],'component_members':components,
            'isolated_atom_count':sum(len(x)==1 for x in components),
            'minimum_pair_distance_A':float(pairs.min()) if pairs.size else None,
            'maximum_pair_distance_A':float(pairs.max()) if pairs.size else None,
            'component_centroids_A':[p[x].mean(axis=0).tolist() for x in components],
            'component_net_forces_eV_per_A':[f[x].sum(axis=0).tolist() for x in components]}


def summarize_trace(trace, cutoff_A):
    rows=[];first_trial=None;first_accepted=None;accepted_graph=None
    start=trace['queries'][0]['positions'].reshape(-1,3)
    max_excursion=0.
    for q in trace['queries']:
        f=q['forces'];p=q['positions'].reshape(-1,3)
        row={k:v for k,v in q.items() if k not in ('positions','forces')}
        if f is not None and f.shape==q['positions'].shape and np.isfinite(f).all():
            f=f.reshape(-1,3);g=graph_summary(p,f,cutoff_A);row.update(g)
            row['max_force_component_eV_per_A']=float(np.max(np.abs(f)))
            row['maximum_force_component_index']=list(np.unravel_index(np.argmax(np.abs(f)),f.shape))
            row['net_force_eV_per_A']=f.sum(axis=0).tolist()
            row['centroid_torque_eV']=np.cross(p-p.mean(axis=0),f).sum(axis=0).tolist()
            parent=q['parent_query']
            delta=np.zeros_like(p) if parent is None else p-trace['queries'][parent]['positions'].reshape(-1,3)
            row['max_abs_displacement_component_A']=float(np.max(np.abs(delta)))
            norms=np.linalg.norm(delta,axis=1)
            row['max_atom_displacement_A']=float(norms.max())
            row['max_displacement_atom']=int(norms.argmax())
            row['component_membership_changed']=accepted_graph is not None and g['component_members']!=accepted_graph['component_members']
            previous={} if accepted_graph is None else {tuple(m):c for m,c in zip(accepted_graph['component_members'],accepted_graph['component_centroids_A'])}
            row['same_component_centroid_displacement_A']=[
                (np.asarray(c)-previous[tuple(m)]).tolist() if tuple(m) in previous else None
                for m,c in zip(g['component_members'],g['component_centroids_A'])]
            disconnected=g['interaction_component_count']>1
            if q['role']=='trial' and disconnected and first_trial is None:first_trial=q['index']
            if q['role']=='start' or q['accepted']:
                if disconnected and first_accepted is None:first_accepted=q['index']
                accepted_graph=g
                max_excursion=max(max_excursion,float(np.max(np.linalg.norm(p-start,axis=1))))
        rows.append(row)
    return {'queries':rows,'first_disconnected_trial':first_trial,
            'first_accepted_disconnected_state':first_accepted,
            'disconnected_at_start':rows[0].get('interaction_component_count',1)>1,
            'maximum_accepted_path_atom_displacement_from_start_A':max_excursion}
