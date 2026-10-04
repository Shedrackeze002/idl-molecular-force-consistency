"""Real tiny numerical trials verify fresh state, actual oracles and child work."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


def runner():return importlib.import_module('force_consistency.capacity_runs')


def tiny_setup():
    from test_coverage import seven_case
    from force_consistency.coverage_runs import coverage_metrics
    from force_consistency.runs import code_provenance
    from force_consistency.late_rate import WEIGHTS
    from dataclasses import asdict
    p,m,b,n,s=seven_case();case={'name':'width4','features':4,'wall_seconds':120}
    c={'cases':[case],'model':{k:v for k,v in asdict(m.config).items() if k!='features'},'seed':0,
       'backend':'cpu','precision':'float64','learning_rate':.0003,'loss_weights':WEIGHTS,
       'microbatch_size':3,'max_updates':3,'score_updates':[0,1,2,3],'oracle_updates':[1,3],
       'gate_fraction':.1,'checkpoint_prediction_atol':1e-10,'finalization_reserve_seconds':.01}
    support={'fitting':p,'anchor':p,'norm':n,'baselines':coverage_metrics(s['params'],p,p,m,3),
             'source_pins':{},'code':code_provenance(Path.cwd()),'source_provenance_sha256':None}
    return c,case,support


def real_tiny_probe(root,timeout,token):
    code='import sys;from force_consistency.capacity_runs import _probe_saved;_probe_saved(sys.argv[1],sys.argv[2],production=False)'
    subprocess.run([sys.executable,'-c',code,str(root.resolve()),token],check=True,capture_output=True,timeout=timeout)
    return json.loads((root/'checkpoint-verification.json').read_text())


def test_fresh_training_matches_independent_multistep_adam_and_real_child(tmp_path):
    import jax
    from force_consistency.train import initialize_state
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.models.energy import EnergyModel
    from force_consistency.metrics import partition_batch
    from force_consistency.loss_balance import weighted_loss
    from force_consistency.width_rate_runs import independent_adam_step
    from force_consistency.runs import load_checkpoint
    from test_coverage import close_trees
    r=runner();c,case,p=tiny_setup();model=EnergyModel(ModelConfig(**c['model'],features=4))
    initial=initialize_state(model,partition_batch(p['anchor'],np.arange(3)),p['norm'],0,.0003)
    wanted=copy.deepcopy(initial);full=partition_batch(p['fitting'],np.arange(7))
    gradient=jax.jit(jax.grad(lambda pp:weighted_loss(pp,full,model,p['norm'],c['loss_weights'])[0]))
    for u in range(1,4):
        pp,oo=independent_adam_step(wanted['params'],wanted['optimizer'],gradient(wanted['params']),.0003)
        wanted={**wanted,'params':pp,'optimizer':oo,'update':u}
    out=r._run_case(c,case,p,tmp_path/'width4',time.perf_counter()+120,probe_runner=real_tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    assert out['updates']==out['persisted_updates']==out['attempted_updates']==3
    assert out['frame_exposures']==21 and not out['pending_update_unknown']
    assert out['scoring_prediction_frames']==56 and out['reference_prediction_frames']==14
    assert out['initialization_frames']==3 and out['initializations']==1
    assert out['child_work']=={'template_initializations':1,'template_initializer_frames':3,'prediction_frames':14,'training_updates':0}
    assert out['checkpoint_verification']['maximum_absolute_error']==0
    assert out['oracles']['3']['incoming_update']==2 and out['oracles']['3']['incoming_nonzero_moments']
    root=tmp_path/'width4';prov=json.loads((root/'provenance.json').read_text())
    got=load_checkpoint(root/'snapshots/3.msgpack',prov,template=initial)
    close_trees((got['params'],got['optimizer']),(wanted['params'],wanted['optimizer']))
    assert out['history'][0]['fitting_gate']['fraction_remaining']['energy_mse_eV2']==pytest.approx(1.)
    with pytest.raises(FileExistsError):r._run_case(c,case,p,root,time.perf_counter()+120,probe_runner=real_tiny_probe)


def test_different_width_uses_its_own_fresh_template(tmp_path):
    r=runner();c,case,p=tiny_setup();case={**case,'name':'width6','features':6};c['cases']=[case]
    out=r._run_case(c,case,p,tmp_path/'width6',time.perf_counter()+120,probe_runner=real_tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    assert out['checkpoint_verification']['case']==case
    assert out['checkpoint_verification']['initial_state_hashes']==out['initial_state_hashes']
    assert out['history'][0]['fitting_gate']['fraction_remaining']['force_mse_eV2_per_A2']==pytest.approx(1.)


def test_unreturned_update_retains_known_computed_and_committed_counts(tmp_path,monkeypatch):
    r=runner();c,case,p=tiny_setup();c.update(max_updates=5,score_updates=[0,1,5],oracle_updates=[1])
    from force_consistency import loss_balance
    real=loss_balance.build_weighted_step
    def factory(*a):
        step=real(*a);calls=0
        def wrapped(*args):
            nonlocal calls
            calls+=1
            if calls==5:raise KeyboardInterrupt('unreturned third training update')
            return step(*args)
        wrapped.prepare=step.prepare;return wrapped
    monkeypatch.setattr(loss_balance,'build_weighted_step',factory)
    out=r._run_case(c,case,p,tmp_path/'width4',time.perf_counter()+120,probe_runner=real_tiny_probe)
    assert not out['execution_valid']
    assert out['updates']==2 and out['attempted_updates']==3 and out['persisted_updates']==1
    assert out['pending_update_unknown'] and out['frame_exposures']==14
    assert json.loads((tmp_path/'width4/commit.json').read_text())['update']==1


def test_orphan_checkpoint_does_not_advance_published_receipt(tmp_path,monkeypatch):
    r=runner();c,case,p=tiny_setup()
    from force_consistency import late_rate
    real=late_rate.atomic_json
    def fail(path,value):
        if Path(path).name=='commit.json' and value['update']==1:raise OSError('publication failed')
        return real(path,value)
    monkeypatch.setattr(late_rate,'atomic_json',fail)
    out=r._run_case(c,case,p,tmp_path/'width4',time.perf_counter()+120,probe_runner=real_tiny_probe)
    assert not out['execution_valid'] and out['updates']==1 and out['persisted_updates']==0
    assert (tmp_path/'width4/snapshots/1.msgpack').exists()
    assert json.loads((tmp_path/'width4/commit.json').read_text())['update']==0


def test_known_child_work_survives_later_source_failure(tmp_path,monkeypatch):
    r=runner();c,case,p=tiny_setup()
    from force_consistency import capacity_contract as cc
    real=cc.verify_pins;calls=0
    def late_failure(pins):
        nonlocal calls
        if not pins:
            calls+=1
            if calls==2:raise ValueError('late source failure')
        return real(pins)
    monkeypatch.setattr(cc,'verify_pins',late_failure)
    out=r._run_case(c,case,p,tmp_path/'width4',time.perf_counter()+120,probe_runner=real_tiny_probe)
    assert not out['execution_valid'] and out['probe_status']=='completed'
    assert out['child_work']['prediction_frames']==14
    assert out['child_work']['template_initializations']==1


def test_invalid_public_config_is_rejected_before_numerical_import(tmp_path):
    c,case,p=tiny_setup();path=tmp_path/'invalid.json';path.write_text(json.dumps(c))
    code="import sys;from force_consistency.capacity_runs import execute\ntry:execute(__import__('json').load(open(sys.argv[1])))\nexcept ValueError:pass\nelse:raise AssertionError('invalid contract accepted')\nassert 'jax' not in sys.modules"
    subprocess.run([sys.executable,'-c',code,str(path)],check=True,capture_output=True)


def test_template_mismatch_cannot_be_silently_restored():
    import jax
    from force_consistency.train import initialize_state
    from force_consistency.models.energy import EnergyModel
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.metrics import partition_batch
    from force_consistency.width_rate_runs import canonical_state_hashes
    r=runner();c,case,p=tiny_setup();model=EnergyModel(ModelConfig(**c['model'],features=4));batch=partition_batch(p['anchor'],np.arange(3))
    first=initialize_state(model,batch,p['norm'],0,.0003);other=initialize_state(model,batch,p['norm'],1,.0003)
    r.verify_template(first,{'initial_state_hashes':canonical_state_hashes(first)})
    with pytest.raises(ValueError):r.verify_template(other,{'initial_state_hashes':canonical_state_hashes(first)})

@pytest.mark.parametrize('failure',['exit','timeout'])
def test_failed_child_retains_stdout_stderr_without_retry(tmp_path,monkeypatch,failure):
    r=runner();root=tmp_path/'child';root.mkdir();calls=[]
    def run(*a,**k):
        calls.append(a)
        if failure=='exit':raise subprocess.CalledProcessError(2,a[0],output=b'child-started',stderr=b'child-failed')
        raise subprocess.TimeoutExpired(a[0],1,output=b'child-started',stderr=b'child-failed')
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises((subprocess.CalledProcessError,subprocess.TimeoutExpired)):r._launch_probe(root,1,'token')
    assert len(calls)==1
    assert (root/'probe-stdout.txt').read_bytes()==b'child-started'
    assert (root/'probe-stderr.txt').read_bytes()==b'child-failed'
