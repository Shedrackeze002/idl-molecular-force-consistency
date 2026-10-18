"""Physical consistency diagnostics and genuine nonzero dual-head checkpoint scoring."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def module():
    assert importlib.util.find_spec('diagnose_e3x_consistency'),'descriptive companion missing'
    return __import__('diagnose_e3x_consistency')


def test_literal_physical_units_residual_sign_and_unequal_group_reductions():
    m=module();f=np.array([3.,1.,1.])[:,None,None]*np.ones((3,2,3));fe=np.ones_like(f);ref=np.zeros_like(f)
    x=m.diagnostic_metrics(['a','b','b'],f,fe,ref,2.)
    assert x['pooled_energy_head_force_mae_eV_per_A']==1.
    assert x['equal_connection_energy_head_force_mae_eV_per_A']==1.
    assert x['residual_mse_eV2_per_A2']==pytest.approx(4/3)
    assert x['residual_rmse_eV_per_A']==pytest.approx((4/3)**.5)
    assert x['pooled_residual_mae_eV_per_A']==pytest.approx(2/3)
    assert x['equal_connection_residual_mae_eV_per_A']==1.
    assert x['normalized_residual_mse']==pytest.approx(1/3)
    assert m.diagnostic_metrics(['a','b','b'],fe,fe,ref,2.)['residual_mse_eV2_per_A2']==0


@pytest.mark.parametrize('scale',[0.,-1.,float('inf'),float('nan'),True])
def test_bad_force_scale_refused(scale):
    m=module();x=np.ones((2,2,3))
    with pytest.raises(ValueError):m.diagnostic_metrics(['a','b'],x,x,x,scale)


def test_nonfinite_and_misaligned_arrays_refused():
    m=module();x=np.ones((2,2,3));bad=x.copy();bad[0,0,0]=np.nan
    for groups,f,fe in [(['a'],x,x),(['a','b'],bad,x),(['a','b'],x,x[:1])]:
        with pytest.raises(ValueError):m.diagnostic_metrics(groups,f,fe,x,1.)


@pytest.mark.parametrize('arm,strength',[('direct',0.),('consistency',1.)])
def test_real_synthetic_dualhead_best_scorer_and_companion(tmp_path,monkeypatch,arm,strength):
    from dataclasses import asdict
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.prepare import export_prepared
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.train import RunConfig,train
    import e3x_phase_common as common
    import evaluate_e3x_adequacy as scorer
    m=module();records,inv,ends=fixture();data=tmp_path/'data'
    manifest=export_prepared(records,directory=data,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    for mod in (common,scorer,m):monkeypatch.setattr(mod,'SPLIT',manifest['split_hash'])
    cfg=common.fixed_config('C',data,tmp_path/'fit',2,180,arm=arm,strength=strength)
    cfg.update(expected_split_hash=manifest['split_hash'],validation_interval=1,model=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero')))
    real_load=np.load;opened=[]
    def guarded(file,*args,**kw):
        name=Path(str(file)).name;assert name!='test.npz','scientific test opened';opened.append(name);return real_load(file,*args,**kw)
    monkeypatch.setattr(np,'load',guarded)
    summary=train(RunConfig(**cfg));assert summary['status']=='passed',summary
    launch={'candidate':arm,'config':cfg,'source_pin':common.source_pin(),'dataset_pin':common.dataset_pin(data),'initial_identity':{k:summary['provenance'][k] for k in ('initial_parameter_sha256','initial_state_sha256')},'external_scripts':m.external_scripts()}
    score=scorer.score_best(tmp_path/'fit',launch,tmp_path/'score')
    assert score['best_update']>0
    with np.load(tmp_path/'score/best-validation.npz') as z:assert np.max(np.abs(z['forces']))>0
    result=m.diagnose_best(tmp_path/'fit',launch,tmp_path/'score',tmp_path/'diagnostic')
    assert result['status']=='verified_descriptive' and result['record_count']==len(manifest['partitions']['validation']['record_ids'])
    assert result['dataset_pin']==launch['dataset_pin']
    assert result['launch_record_sha256']==common.digest(common.encoded(launch))
    with np.load(tmp_path/'diagnostic/consistency-validation.npz') as z:
        assert z['record_ids'].tolist()==manifest['partitions']['validation']['record_ids']
        np.testing.assert_array_equal(z['residual'],z['direct_forces']-z['energy_head_forces'])
    assert 'test.npz' not in opened
    launch['external_scripts']['diagnose_e3x_consistency.py']='changed'
    with pytest.raises(ValueError,match='External'):m.diagnose_best(tmp_path/'fit',launch,tmp_path/'score',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
