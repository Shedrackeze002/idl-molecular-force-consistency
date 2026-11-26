"""Synthetic observation-neutrality checks; no trained model execution."""
import importlib.util
import json
import time

import numpy as np
import pytest

from evaluation.relax import relax
from evaluation.adapter import NonfiniteValueError


def module():
    assert importlib.util.find_spec('recovery_trace'), 'Trace helper missing'
    return __import__('recovery_trace')


P = dict(max_force_evaluations=6, initial_step_size=.2,
         backtracking_factor=.5, min_backtracking_floor=1e-8,
         convergence_tolerance_eV_per_A=.01,
         max_backtracking_attempts_per_step=3)
START = np.array([[0., 0., 0.], [1., 0., 0.]])


def supplier(sequence):
    values = iter(sequence)
    calls = []
    def fn(p):
        calls.append(p.copy())
        value = next(values)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return value
        if np.isscalar(value):
            return np.full_like(p, value)
        return value.copy()
    return fn, calls


def identical(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            identical(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    else:
        assert a == b


@pytest.mark.parametrize('values,overrides', [
    ([0.], {}),
    ([1., -.001], {}),  # convergence precedes a negative agreement
    ([1., -1., -1., .5, 0.], {}),
    ([1.], {'max_force_evaluations': 1}),
    ([1., -1.], {'max_force_evaluations': 2}),
    ([1., -1., -1., -1.], {}),
    ([1e-10], {'convergence_tolerance_eV_per_A': 1e-12}),
    ([1., -1., -1., -1.], {'min_backtracking_floor': .1}),
    ([1., np.array([0.])], {}),
    ([1., 'bad'], {}),
    ([1., np.array([[[np.nan]*3]*2])], {}),
])
def test_wrapper_is_bitwise_neutral(tmp_path, values, overrides):
    m = module(); procedure = P | overrides
    plain, a = supplier(values); observed, b = supplier(values)
    wrapped = m.TraceSupplier(observed, tmp_path/'trace', 1500,
                              time.perf_counter()+30, 2**24)
    expected = relax(START, plain, procedure)
    actual = relax(START, wrapped, procedure)
    identical(expected, actual)
    identical(a, b)
    assert len(wrapped.queries) == actual['force_evaluation_count']
    reconstructed = m.reconstruct_trace(wrapped.queries, actual, procedure)
    assert reconstructed['status'] == 'verified'
    assert sum(q['accepted'] for q in reconstructed['queries']) == actual['step_count']
    journal = [json.loads(x) for x in (tmp_path/'trace/queries.jsonl').read_text().splitlines()]
    assert sum(x['event'] == 'attempted' for x in journal) == len(a)


def test_original_return_object_and_reused_array_are_preserved(tmp_path):
    m = module(); array = np.ones((1,2,3)); seen = []
    def fn(p):
        seen.append(p.copy()); return array
    wrapped = m.TraceSupplier(fn, tmp_path/'t', 3, time.perf_counter()+30, 2**24)
    assert wrapped(START[None]) is array
    array[:] = 2
    assert np.all(wrapped.queries[0]['forces'] == 1)
    assert len(seen) == 1


@pytest.mark.parametrize('exception', [ValueError('internal'), NonfiniteValueError('bad force', {})])
def test_exception_identity_and_durable_attempt(tmp_path, exception):
    m = module()
    def fn(p):
        journal = (tmp_path/'t/queries.jsonl').read_text()
        assert 'attempted' in journal
        raise exception
    wrapped = m.TraceSupplier(fn, tmp_path/'t', 3, time.perf_counter()+30, 2**24)
    with pytest.raises(type(exception)) as caught:
        wrapped(START[None])
    assert caught.value is exception
    assert wrapped.queries[0]['state'] == 'raised'


@pytest.mark.parametrize('limit,deadline', [(0,time.perf_counter()+30), (3,0)])
def test_caps_prevent_supplier_invocation(tmp_path, limit, deadline):
    m = module(); calls = []
    wrapped = m.TraceSupplier(lambda p:calls.append(p), tmp_path/'t', limit, deadline, 2**24)
    with pytest.raises(RuntimeError):
        wrapped(START[None])
    assert calls == []


def test_storage_reservation_and_no_repeat(tmp_path):
    m = module(); calls = []
    wrapped = m.TraceSupplier(lambda p:calls.append(p), tmp_path/'t', 3, time.perf_counter()+30, 100)
    with pytest.raises(RuntimeError):
        wrapped(START[None])
    assert calls == []
    with pytest.raises(FileExistsError):
        m.TraceSupplier(lambda p:p, tmp_path/'t', 3, time.perf_counter()+30, 2**24)


def test_reconstruction_rejects_modified_trial(tmp_path):
    m = module(); fn,_ = supplier([1.,0.]);w=m.TraceSupplier(fn,tmp_path/'t',3,time.perf_counter()+30,2**24)
    result=relax(START,w,P);w.queries[1]['positions'][0,0,0] += .01
    with pytest.raises(ValueError,match='candidate'):
        m.reconstruct_trace(w.queries,result,P)


def test_graph_and_displacement_component_vs_vector(tmp_path):
    m=module();fn,_=supplier([1.,0.]);w=m.TraceSupplier(fn,tmp_path/'t',3,time.perf_counter()+30,2**24)
    result=relax(START,w,P);t=m.reconstruct_trace(w.queries,result,P);s=m.summarize_trace(t,5.)
    q=s['queries'][1]
    assert q['max_abs_displacement_component_A'] == pytest.approx(.2)
    assert q['max_atom_displacement_A'] == pytest.approx(np.sqrt(3)*.2)
    assert q['interaction_component_count'] == 1
    assert q['component_members'] == [[0,1]]
    assert s['first_disconnected_trial'] is None


def test_disconnected_at_start_and_rejected_trial_are_distinct(tmp_path):
    m=module();start=START.copy();start[1,0]=6.
    fn,_=supplier([0.]);w=m.TraceSupplier(fn,tmp_path/'t',3,time.perf_counter()+30,2**24)
    result=relax(start,w,P);s=m.summarize_trace(m.reconstruct_trace(w.queries,result,P),5.)
    assert s['disconnected_at_start'] is True
    assert s['queries'][0]['isolated_atom_count'] == 2
    assert s['first_accepted_disconnected_state'] == 0


def test_changed_member_partition_not_compared_as_same_fragment():
    m=module()
    a=m.graph_summary(np.array([[0.,0,0],[1.,0,0],[9.,0,0]]),np.ones((3,3)),5.)
    b=m.graph_summary(np.array([[0.,0,0],[8.,0,0],[9.,0,0]]),np.ones((3,3)),5.)
    assert a['component_members'] == [[0,1],[2]]
    assert b['component_members'] == [[0],[1,2]]


def test_last_rejection_trial_limit_takes_precedence_over_budget(tmp_path):
    m=module();fn,_=supplier([1.,-1.,-1.,-1.]);p=P|{'max_force_evaluations':4}
    w=m.TraceSupplier(fn,tmp_path/'t',4,time.perf_counter()+30,2**24)
    out=relax(START,w,p)
    assert out['failure_reason']=='no_progress'
    assert m.reconstruct_trace(w.queries,out,p)['status']=='verified'


def test_rejected_disconnected_trial_is_not_accepted_separation(tmp_path):
    m=module();f=np.array([[[-10.,0,0],[10.,0,0]]])
    fn,_=supplier([f,-f,np.zeros_like(f)])
    w=m.TraceSupplier(fn,tmp_path/'t',3,time.perf_counter()+30,2**24)
    out=relax(START,w,P);s=m.summarize_trace(m.reconstruct_trace(w.queries,out,P),5.)
    assert s['first_disconnected_trial']==1
    assert s['first_accepted_disconnected_state'] is None
    assert s['queries'][1]['decision']=='rejected'


def test_admission_journal_failure_never_calls_supplier(tmp_path,monkeypatch):
    m=module();calls=[];w=m.TraceSupplier(lambda p:calls.append(p),tmp_path/'t',3,time.perf_counter()+30,2**24)
    def broken(event):raise OSError('disk error')
    monkeypatch.setattr(w,'_event',broken)
    with pytest.raises(OSError):w(START[None])
    assert calls==[] and w.budget.invocations==0


def test_new_wrapper_does_not_reset_shared_query_cap(tmp_path):
    m=module();b=m.TraceBudget(1,time.perf_counter()+30,2**24)
    first=m.TraceSupplier(lambda p:np.zeros_like(p),tmp_path/'a',3,b.deadline,b.max_bytes,budget=b)
    first(START[None]);calls=[]
    second=m.TraceSupplier(lambda p:calls.append(p),tmp_path/'b',3,b.deadline,b.max_bytes,budget=b)
    with pytest.raises(RuntimeError):second(START[None])
    assert calls==[] and b.invocations==1


def test_rejected_tail_checks_budget_before_outer_small_step(tmp_path):
    m=module();p=P|{'max_force_evaluations':2,'initial_step_size':1e-8,'min_backtracking_floor':1e-10,
                  'convergence_tolerance_eV_per_A':1e-12}
    fn,_=supplier([.15,-.15]);w=m.TraceSupplier(fn,tmp_path/'t',2,time.perf_counter()+30,2**24)
    out=relax(START,w,p)
    assert out['failure_reason']=='budget_exhausted'
    assert m.reconstruct_trace(w.queries,out,p)['status']=='verified'


def test_convergence_trial_logs_negative_agreement(tmp_path):
    m=module();fn,_=supplier([1.,-.001]);w=m.TraceSupplier(fn,tmp_path/'t',2,time.perf_counter()+30,2**24)
    out=relax(START,w,P);q=m.reconstruct_trace(w.queries,out,P)['queries'][1]
    assert q['decision']=='converged_on_trial' and q['accepted']
    assert q['agreement']==pytest.approx(-.006)


def test_admission_refusal_retains_relax_attempt_without_supplier_call(tmp_path):
    m=module();w=m.TraceSupplier(lambda p:p,tmp_path/'t',0,time.perf_counter()+30,2**24)
    with pytest.raises(RuntimeError):w(START[None])
    assert w.budget.relax_attempts==1 and w.budget.invocations==0
    assert w.admission_failures[0]['supplier_invoked'] is False
    assert 'admission_refused' in (tmp_path/'t/queries.jsonl').read_text()


def test_logging_failure_after_invocation_keeps_raw_return(tmp_path,monkeypatch):
    m=module();w=m.TraceSupplier(lambda p:np.zeros_like(p),tmp_path/'t',3,time.perf_counter()+30,2**24)
    original=w._event
    def fail_after(event):
        if event['event']=='returned':raise OSError('post-call write failure')
        original(event)
    monkeypatch.setattr(w,'_event',fail_after)
    with pytest.raises(OSError):w(START[None])
    assert w.budget.invocations==1
    assert w.queries[0]['state']=='returned'
    assert np.array_equal(w.queries[0]['forces'],np.zeros_like(START[None]))


def test_supplier_reusing_mutating_return_array_does_not_change_relaxation(tmp_path):
    m=module()
    def create():
        array=np.ones((1,2,3));values=iter([1.,.5,0.])
        def fn(p):array[:]=next(values);return array
        return fn
    expected=relax(START,create(),P)
    w=m.TraceSupplier(create(),tmp_path/'t',3,time.perf_counter()+30,2**24)
    actual=relax(START,w,P)
    identical(actual,expected)
