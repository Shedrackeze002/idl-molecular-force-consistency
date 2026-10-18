"""Genuine synthetic checkpoint supplier replay; never scientific training."""
import copy
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def adapter():
    assert importlib.util.find_spec('evaluation.e3x_adapter'), 'E3x supplier missing'
    return __import__('evaluation.e3x_adapter',fromlist=['e3x_checkpoint_arm'])


@pytest.mark.parametrize('arm,strength',[('energy',0.),('direct',0.),('consistency',.1)])
def test_genuine_synthetic_checkpoint_supplier_replay_and_refusals(tmp_path,monkeypatch,arm,strength,synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification):
    m=adapter()
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.prepare import export_prepared,load_training_bundle
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.train import train,RunConfig
    from evaluation.adapter import ArmConfigurationError
    import e3x_phase_common as common
    import evaluate_e3x_adequacy as scorer
    records,inventory,ends=fixture();positions=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]])
    forces=np.array([[-1.,-1.,0],[1.,0,0],[0.,1.,0]])
    records=[replace(r,species=('C','H','H'),positions=positions,forces=forces) for r in records]
    data=tmp_path/'data';manifest=export_prepared(records,directory=data,inventory=inventory,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    for module in [common,scorer]:monkeypatch.setattr(module,'SPLIT',manifest['split_hash'])
    cfg=common.fixed_config('C',data,tmp_path/'fit',2,180,arm=arm,strength=strength)
    cfg.update(expected_split_hash=manifest['split_hash'],validation_interval=1,model=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero')))
    real_load=np.load;opened=[]
    def guarded(file,*args,**kw):
        assert Path(str(file)).name!='test.npz';opened.append(str(file));return real_load(file,*args,**kw)
    monkeypatch.setattr(np,'load',guarded)
    summary=train(RunConfig(**cfg));assert summary['status']=='passed',summary
    launch={'candidate':arm,'config':cfg,'source_pin':common.source_pin(),'dataset_pin':common.dataset_pin(data),'initial_identity':{k:summary['provenance'][k] for k in ['initial_parameter_sha256','initial_state_sha256']}}
    score=scorer.score_best(tmp_path/'fit',launch,tmp_path/'score')
    assert score['best_update']>0, 'Synthetic supplier fixture must exercise learned weights'
    binding={'checkpoint_sha256':score['best_checkpoint_sha256'],'best_update':score['best_update'],'score_sha256':common.digest((tmp_path/'score/score.json').read_bytes())}
    for key in binding:
        bad=copy.deepcopy(binding);bad[key]='changed'
        with pytest.raises(ValueError):m.e3x_checkpoint_arm(tmp_path/'fit',launch,tmp_path/'score',acceptance=bad)
    wrong=copy.deepcopy(launch);wrong['source_pin']['code_fingerprint']='changed'
    with pytest.raises(ValueError):m.e3x_checkpoint_arm(tmp_path/'fit',wrong,tmp_path/'score',acceptance=binding)
    supplier=m.e3x_checkpoint_arm(tmp_path/'fit',launch,tmp_path/'score',acceptance=binding)
    assert supplier['status']=='ready' and supplier['species']==('C','H','H')
    assert supplier['checkpoint_identity']['checkpoint_sha256']==binding['checkpoint_sha256']
    bundle=load_training_bundle(data,expected_split_hash=manifest['split_hash'])
    with np.load(tmp_path/'score/best-validation.npz') as z:
        expected=z['forces'][:4];expected_energy=float(z['energy'][0])
    f,info=supplier['forces'](bundle.validation.positions[:4],species=('C','H','H'),record_id='synthetic-replay')
    np.testing.assert_allclose(f,expected,rtol=1e-10,atol=1e-10)
    single,_=supplier['forces'](positions,species=('C','H','H'))
    assert single.dtype==np.float64 and single.shape==(3,3) and np.max(np.abs(single))>0
    np.testing.assert_allclose(single,f[0],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(supplier['forces'](positions+np.array([2.,-3.,4.]))[0],single,rtol=1e-9,atol=1e-9)
    rotation=np.array([[0.,-1,0],[1.,0,0],[0.,0,1.]])
    np.testing.assert_allclose(supplier['forces'](positions@rotation)[0],single@rotation,rtol=1e-9,atol=1e-9)
    permutation=[0,2,1]
    np.testing.assert_allclose(supplier['forces'](positions[permutation])[0],single[permutation],rtol=1e-9,atol=1e-9)
    if arm=='energy':
        assert supplier['provides_energy'] is True
        energy,_=supplier['energy'](positions);assert energy==pytest.approx(expected_energy,abs=1e-10)
        assert supplier['checkpoint_identity']['energy_convention']=='training-offset residual eV'
    else:
        assert supplier['provides_energy'] is False
        with pytest.raises(LookupError):supplier['energy'](positions)
    calls=[];actual=m.make_batch
    monkeypatch.setattr(m,'make_batch',lambda *args,**kw:(calls.append(1),actual(*args,**kw))[1])
    with pytest.raises(ArmConfigurationError):supplier['forces'](positions,species=('H','C','H'))
    with pytest.raises(ValueError):supplier['forces'](np.ones((3,2)))
    with pytest.raises(ValueError):supplier['forces'](positions*np.nan)
    with pytest.raises(ValueError):supplier['forces'](np.empty((0,3,3)))
    assert not calls
    checkpoint=tmp_path/'fit/best.msgpack';original=checkpoint.read_bytes();checkpoint.write_bytes(original+b'changed')
    with pytest.raises(ValueError):m.e3x_checkpoint_arm(tmp_path/'fit',launch,tmp_path/'score',acceptance=binding)
    checkpoint.write_bytes(original)
    assert all(not x.endswith('test.npz') for x in opened)

    # Inject predictor OUTPUTS into a genuine restored supplier, not input NaNs.
    import jax
    from evaluation.adapter import NonfiniteValueError
    import run_validation_recovery as driver
    from test_validation_recovery import synthetic_pass_inputs
    with monkeypatch.context() as patch:
        patch.setattr(jax,'device_get',lambda _: (np.array([np.nan]),np.full((1,3,3),np.nan)))
        with pytest.raises(NonfiniteValueError):supplier['forces'](positions,record_id='nonfinite-output')
        inputs=list(synthetic_pass_inputs(synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification))
        inputs[0]={k:{**v,'species':supplier['species']} for k,v in inputs[0].items()}
        inputs[1]=supplier
        inputs[4]={k:replace(v,species=supplier['species']) for k,v in inputs[4].items()}
        result=driver.run_model_pass(*inputs,seen_minimum=False,output=tmp_path/'nonfinite-pass')
        rows=list((tmp_path/'nonfinite-pass/raw-branches').glob('*.json'))
        assert len(rows)==len(inputs[0])>1 and result['complete']
        for path in rows:
            row=json.loads(path.read_text())
            assert row['force_evaluation_count']==1 and row['category']=='nonconverged'
            assert 'nonfinite_force' in json.dumps(row) and not row['converged']
    with monkeypatch.context() as patch:
        patch.setattr(jax,'device_get',lambda _: (np.array([np.nan]),np.ones((1,3,3))))
        np.testing.assert_array_equal(supplier['forces'](positions)[0],np.ones((3,3)))
        if arm=='energy':
            with pytest.raises(NonfiniteValueError):supplier['energy'](positions,record_id='nonfinite-energy')
