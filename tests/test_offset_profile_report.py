"""Saved-only audit detects receipt swaps, orphan publication and false endorsement."""
import importlib.util
from pathlib import Path
import sys
import pytest

def api():
    name='report_offset_profile';s=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts/report_offset_profile.py')
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def test_saved_tiny_run_recomputes_metrics_without_model_evaluation(tmp_path,monkeypatch):
    from test_offset_profile_runs import small_run
    m,c,source,gate,root,result=small_run(tmp_path);assert result['execution_valid'],result
    from force_consistency.models.energy import EnergyModel
    def forbidden(*a,**k):raise AssertionError('Numerical model invoked in saved-only audit')
    monkeypatch.setattr(EnergyModel,'apply',forbidden)
    report=api().audit_saved(root,complete=False,source=source,gate=gate)
    assert report['scored_states']==3 and report['accepted_projection_receipts']==4
    assert report['recomputed_endpoint']==result['endpoint']['metrics']

@pytest.mark.parametrize('mutation',['ids','projection','orphan','biasmoment','rng'])
def test_saved_audit_rejects_source_or_receipt_substitution(tmp_path,mutation):
    from test_offset_profile_runs import small_run
    m,c,source,gate,root,result=small_run(tmp_path);assert result['execution_valid'],result
    if mutation=='ids':
        p=root/'projections/training-2.json';r=m.cc.read(p);r['fitting_record_ids']=r['fitting_record_ids'][::-1];m.cc.atomic(p,r)
    elif mutation=='projection':
        p=root/'accepted/2.json';r=m.cc.read(p);r['projection']='projections/training-1.json';r['projection_sha256']=m.cc.digest(root/r['projection']);m.cc.atomic(p,r)
    elif mutation=='orphan':
        (root/'commits/3.json').unlink()
    else:
        from offset_profile_core import restore_state,set_bias
        from force_consistency.runs import save_checkpoint
        p=m.cc.read(root/'provenance.json');state=restore_state(root/'snapshots/3.msgpack',p)
        if mutation=='biasmoment':state['optimizer']=(state['optimizer'][0]._replace(mu=set_bias(state['optimizer'][0].mu,[.1])),state['optimizer'][1])
        else:state['sampler_rng_json']='{}'
        save_checkpoint(root/'snapshots/3.msgpack',state,p)
        # Update superficial hash to ensure the audit independently checks state fields.
        r=m.cc.read(root/'commit.json');r['checkpoint_sha256']=m.cc.digest(root/'snapshots/3.msgpack');m.cc.atomic(root/'commit.json',r);m.cc.atomic(root/'commits/3.json',r)
    with pytest.raises((ValueError,FileNotFoundError)):api().audit_saved(root,complete=False,source=source,gate=gate)

def test_missing_summary_never_endorses_observations(tmp_path):
    report=api().incomplete_report(tmp_path)
    assert not report['execution_valid'] and not report['scientific_endorsement']

def test_matching_grid_ignores_unmatched_dense_events():
    m=api();a=[{'update':0,'x':9},{'update':1,'x':2},{'update':100,'x':3}]
    b=[{'update':0,'x':0},{'update':1,'x':1},{'update':2,'x':999},{'update':100,'x':2}]
    paired=m.matching_rows(a,b)
    assert [u for u,_,_ in paired]==[1,100]
