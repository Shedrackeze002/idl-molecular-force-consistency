"""Checkpoint failures must not silently restart or change selection."""
import json
from pathlib import Path
import numpy as np
import pytest
from flax import serialization
from force_consistency.runs import save_checkpoint
from experiments.representation_study.runner import inspect_checkpoint,newest_valid_checkpoint,retain_scheduled,IdentityError,completed_segment


def checkpoint(folder,update=1000,best_update=500,score=.2,provenance=None):
    folder.mkdir(parents=True,exist_ok=True);prov={'source':'fixed'} if provenance is None else provenance
    (folder/'provenance.json').write_text(json.dumps(prov))
    state={'params':{'x':np.ones(2,dtype=np.float64)},'optimizer':{'m':np.zeros(2)},'update':update,'sampler_rng_json':'{}','model_rng':np.array([1,2],dtype=np.uint32),'normalization_json':'{}'}
    history=[{'update':u,'equal_connection_force_mae_eV_per_A':score if u==best_update else .5} for u in range(0,update+1,500)]
    if update%500:history.append({'update':update,'equal_connection_force_mae_eV_per_A':score})
    control={'best_score':score,'best_update':best_update,'best_state':serialization.to_bytes({**state,'update':best_update}),'validation_history':history}
    save_checkpoint(folder/'last.msgpack',state,prov,run_control=control);return state,control


def test_corrupt_newest_falls_back_but_wrong_source_stops(tmp_path):
    root=tmp_path/'segments';a=root/'000001';b=root/'000002'
    checkpoint(a);checkpoint(b,1500);(b/'last.msgpack').write_bytes(b'partial')
    selected,rejected=newest_valid_checkpoint(root,{'source':'fixed'})
    assert selected['update']==1000 and rejected and '000002' in rejected[0]['path']
    checkpoint(b,1500,provenance={'source':'changed'})
    with pytest.raises(IdentityError):newest_valid_checkpoint(root,{'source':'fixed'})


def test_off_grid_best_cannot_replace_scheduled_copy(tmp_path):
    root=tmp_path/'segments';s=root/'000001';checkpoint(s)
    assert retain_scheduled(s,{'source':'fixed'})
    checkpoint(s,update=1200,best_update=1200,score=.01)
    assert not retain_scheduled(s,{'source':'fixed'})
    selected,rejected=newest_valid_checkpoint(root,{'source':'fixed'})
    assert selected['update']==1000 and selected['control']['best_update']==500 and rejected
    assert (s/'last.msgpack').exists()


def test_future_control_and_off_grid_only_are_rejected(tmp_path):
    root=tmp_path/'segments';s=root/'000001';checkpoint(s,1200,1200)
    with pytest.raises(ValueError,match='No eligible'):newest_valid_checkpoint(root,{'source':'fixed'})
    state,control=checkpoint(s);control['validation_history'].append({'update':1500,'equal_connection_force_mae_eV_per_A':.3})
    save_checkpoint(s/'last.msgpack',state,{'source':'fixed'},run_control=control)
    with pytest.raises(ValueError,match='schedule'):inspect_checkpoint(s/'last.msgpack',{'source':'fixed'})


def test_completed_skip_requires_reload_proofs_and_verified_best(tmp_path):
    s=tmp_path/'segments/000001';state,control=checkpoint(s,2000)
    save_checkpoint(s/'best.msgpack',{**state,'update':500},{'source':'fixed'},run_control={**control,'validation_history':control['validation_history'][:2]})
    summary={'status':'passed','updates':2000,'stop_reason':'max_updates','provenance':{'source':'fixed'},'best_update':500}
    (s/'summary.json').write_text(json.dumps(summary))
    with pytest.raises((ValueError,FileNotFoundError)):completed_segment(s,{'source':'fixed'},2000)
    for role,u,n in [('last',2000,'checkpoint-verification.json'),('best',500,'best-checkpoint-verification.json')]:
        (s/n).write_text(json.dumps({'status':'passed','role':role,'update':u}))
    assert completed_segment(s,{'source':'fixed'},2000)['update']==2000
    summary['status']='failed';(s/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='complete'):completed_segment(s,{'source':'fixed'},2000)


def test_new_worker_receives_empty_run_directory(tmp_path,monkeypatch):
    import experiments.representation_study.runner as runner
    monkeypatch.setattr(runner,'runtime_identity',lambda c:{'source':'fixed'})
    class Worker:
        pid=999
        returncode=0
        def poll(self):return 0
    def spawn(command,**kwargs):
        config=json.loads(Path(command[-1]).read_text());segment=Path(config['run_directory'])
        assert not list(segment.iterdir()),'Frozen train requires an empty run directory'
        state,control=checkpoint(segment,2000)
        save_checkpoint(segment/'best.msgpack',{**state,'update':500},{'source':'fixed'},run_control={**control,'validation_history':control['validation_history'][:2]})
        (segment/'summary.json').write_text(json.dumps({'status':'passed','updates':2000,'stop_reason':'max_updates','provenance':{'source':'fixed'},'best_update':500}))
        for role,u,name in [('last',2000,'checkpoint-verification.json'),('best',500,'best-checkpoint-verification.json')]:
            (segment/name).write_text(json.dumps({'status':'passed','role':role,'update':u}))
        return Worker()
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    result=runner.run_fit({'max_updates':2000},tmp_path/'newfit')
    assert result['status']=='completed'
    result=runner.run_fit({'max_updates':2000},tmp_path/'newfit')
    assert result['skipped'] and len(list((tmp_path/'newfit/segments').iterdir()))==1


@pytest.mark.parametrize('corruption',['summary','last'])
def test_corrupt_completed_summary_reaches_checkpoint_fallback(tmp_path,monkeypatch,corruption):
    import experiments.representation_study.runner as runner
    monkeypatch.setattr(runner,'runtime_identity',lambda c:{'source':'fixed'})
    fit=tmp_path/'fit';old=fit/'segments/000001';checkpoint(old,1000);retain_scheduled(old,{'source':'fixed'})
    if corruption=='summary':(old/'summary.json').write_text('{truncated')
    else:
        checkpoint(old,2000);(old/'last.msgpack').write_bytes(b'corrupt complete payload')
        (old/'summary.json').write_text(json.dumps({'status':'passed','updates':2000,'stop_reason':'max_updates','provenance':{'source':'fixed'},'best_update':500}))
    class Worker:
        pid=999
        returncode=0
        def poll(self):return 0
    def spawn(command,**kwargs):
        cfg=json.loads(Path(command[-1]).read_text());assert cfg['resume_checkpoint']
        folder=Path(cfg['run_directory']);state,control=checkpoint(folder,2000)
        save_checkpoint(folder/'best.msgpack',{**state,'update':500},{'source':'fixed'},run_control={**control,'validation_history':control['validation_history'][:2]})
        (folder/'summary.json').write_text(json.dumps({'status':'passed','updates':2000,'stop_reason':'max_updates','provenance':{'source':'fixed'},'best_update':500}))
        for role,u,name in [('last',2000,'checkpoint-verification.json'),('best',500,'best-checkpoint-verification.json')]:
            (folder/name).write_text(json.dumps({'status':'passed','role':role,'update':u}))
        return Worker()
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    assert runner.run_fit({'max_updates':2000},fit)['status']=='completed'


def test_reference_labels_bind_actual_seed_and_model(tmp_path):
    from experiments.representation_study.protocol import make_configs
    from experiments.representation_study.runner import validate_references
    configs=make_configs('/data',tmp_path,88);paths={}
    for label in ('E64-0','E64-1'):
        folder=tmp_path/label;folder.mkdir();(folder/'config.json').write_text(json.dumps(configs['E64-0']));paths[label]=str(folder)
    with pytest.raises(IdentityError,match='reference'):
        validate_references({'reference_runs':paths},configs)
