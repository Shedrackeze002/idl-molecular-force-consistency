import copy
import json
import shutil
from pathlib import Path
import pytest
from force_consistency import optimizer_runs as runs
from force_consistency.coverage_runs import _state_hashes,WallBudget
from force_consistency.diagnostics import sha256
from force_consistency.data.prepare import write_json
from test_gradient_diagnostics import witness


def config():return json.loads(Path('configs/aspirin-optimizer-step-diagnostic.json').read_text())


@pytest.mark.parametrize('key,value',[('fd_atol',1.),('fd_rtol',1.),('gradient_atol',1.),('gradient_rtol',1.),('fd_steps',[.1]),('probe_scales',[.5]),('directions',['steepest_matched_norm']),('cases',['balanced']),('diagnostic_wall_seconds',601),('adam',{'b1':.8}),('replay_tolerance',{'atol':1.,'rtol':1.})])
def test_settings_rejected_before_parent_loader(monkeypatch,key,value):
    c=config();c[key]=value
    monkeypatch.setattr(runs,'balance_preflight',lambda c:pytest.fail('array loader reached'))
    with pytest.raises(ValueError):runs.optimizer_preflight(c)


def test_omitted_and_stale_pins_before_arrays(monkeypatch):
    monkeypatch.setattr(runs,'balance_preflight',lambda c:pytest.fail('array loader reached'))
    for key in ('loss_balance_sha256','source_sha256'):
        c=config();c[key].pop(next(iter(c[key])))
        with pytest.raises(ValueError):runs.optimizer_preflight(c)
    c=config();c['loss_balance_compact_sha256']='0'*64
    with pytest.raises(ValueError):runs.optimizer_preflight(c)


def test_rehashed_wrong_metadata_before_parent(tmp_path,monkeypatch):
    c=config();root=tmp_path/'source';shutil.copytree(c['loss_balance_run'],root);c['loss_balance_run']=str(root)
    # A matching byte pin alone cannot validate an incorrect source counter.
    d=json.loads((root/'balanced/summary.json').read_text());d['updates']=1999;write_json(root/'balanced/summary.json',d)
    c['loss_balance_sha256']['balanced/summary.json']=sha256(root/'balanced/summary.json')
    compact=json.loads(Path(c['loss_balance_compact_path']).read_text());compact['artifacts_sha256']=c['loss_balance_sha256'];compact['arms']['balanced']=d
    write_json(tmp_path/'compact.json',compact);c['loss_balance_compact_path']=str(tmp_path/'compact.json');c['loss_balance_compact_sha256']=sha256(tmp_path/'compact.json')
    monkeypatch.setattr(runs,'balance_preflight',lambda c:pytest.fail('array loader reached'))
    with pytest.raises(ValueError):runs.optimizer_preflight(c)


def test_real_preflight_immutable_sources_and_counts():
    c=config();fit,anchor,m,n,states,source=runs.optimizer_preflight(c)
    assert len(fit.record_ids)==128 and len(anchor.record_ids)==16
    assert source['compute']['completed_frame_evaluations']['parent_source_predictions']==208
    assert source['compute']['completed_frame_evaluations']['loss_source_predictions']==432
    assert source['raw_source_count']==100 and source['compact_source_count']==2
    for name,s in states.items():
        assert int(s['update'])==2000 and int(s['optimizer'][0].count)==2000
        assert _state_hashes(s)==source['arms'][name]['state_hashes']
    runs.verify_optimizer_sources(c)


@pytest.mark.parametrize('kind',['invalid','exception','budget'])
def test_case_failure_preserves_partial_blocks_remaining(tmp_path,kind):
    part,m,b,n,s=witness();before=_state_hashes(s);c=config();c.pop('evidence_log');states={name:s for name in runs.CASES}
    source={'arms':{name:{'initial_metrics':{}} for name in runs.CASES}}
    called=[]
    def analyze(state,fit,anchor,model,norm,weights,local):
        called.append(weights);local['_save_partial']({'compute':{'operations':[{'status':'started','completed_frames':None}]},'counts':{'fd_cells':0,'fd_steps':0,'probes':0}})
        if kind=='exception':raise RuntimeError('probe failed')
        if kind=='budget':raise WallBudget('budget')
        return {'verification_valid':False,'compute':{'operations':[]},'counts':{'fd_cells':0,'fd_steps':0,'probes':0}}
    result=runs.execute_optimizer_cases(tmp_path,states,part,part,m,n,source,c,lambda:None,analyzer=analyze,verify=lambda:None)
    assert len(called)==1 and result['replay']['status']=='failed'
    assert result['balanced']['status']==result['force-only']['status']=='blocked'
    assert (tmp_path/'replay/partial.json').exists() and _state_hashes(s)==before


def test_protected_root_preflight_failure_and_cli(tmp_path,monkeypatch):
    c=config();c.pop('evidence_log');c['run_directory']=str(tmp_path/'run');c['fd_atol']=1
    result=runs.diagnose_optimizer_steps(c)
    assert not result['execution_valid'] and result['next_step']=='resolve_verification_or_operational_failure'
    assert all(not x['executed'] for x in result['cases'].values()) and result['full_pilot']=='pending'
    with pytest.raises(ValueError):runs.diagnose_optimizer_steps(c)
    write_json(tmp_path/'config.json',c)
    monkeypatch.setattr(runs,'diagnose_optimizer_steps',lambda c:result)
    from force_consistency.cli import main
    assert main(['diagnose-optimizer-step','--config',str(tmp_path/'config.json')])==1


def cloned_sources(tmp_path):
    c=config();root=tmp_path/'source';shutil.copytree(c['loss_balance_run'],root);c['loss_balance_run']=str(root)
    compact=json.loads(Path(c['loss_balance_compact_path']).read_text());return c,root,compact


def repin(c,root,compact,tmp_path,file):
    c['loss_balance_sha256'][file]=sha256(root/file);compact['artifacts_sha256']=c['loss_balance_sha256']
    write_json(tmp_path/'compact.json',compact);c['loss_balance_compact_path']=str(tmp_path/'compact.json');c['loss_balance_compact_sha256']=sha256(tmp_path/'compact.json')


def test_rehashed_wrong_arm_weights_before_parent(tmp_path,monkeypatch):
    c,root,compact=cloned_sources(tmp_path);d=json.loads((root/'balanced/config.json').read_text());d['loss_weights']['force']=1.
    write_json(root/'balanced/config.json',d);repin(c,root,compact,tmp_path,'balanced/config.json')
    monkeypatch.setattr(runs,'balance_preflight',lambda c:pytest.fail('array loader reached'))
    with pytest.raises(ValueError,match='arm configuration'):runs.optimizer_preflight(c)


def test_rehashed_wrong_saved_arrays_rejected(tmp_path):
    import numpy as np
    c,root,compact=cloned_sources(tmp_path);file='replay/fitting-inputs.npz'
    with np.load(root/file) as saved:d={k:saved[k].copy() for k in saved.files}
    d['positions'][0,0,0]+=.01;np.savez(root/file,**d);repin(c,root,compact,tmp_path,file)
    with pytest.raises(ValueError,match='input arrays'):runs.optimizer_preflight(c)


def test_loaded_bad_optimizer_counter_rejected(monkeypatch):
    import jax.numpy as jnp
    original=runs.load_checkpoint
    def bad(*args,**kwargs):
        s=original(*args,**kwargs);s['optimizer']=(s['optimizer'][0]._replace(count=jnp.array(1999,jnp.int32)),s['optimizer'][1]);return s
    monkeypatch.setattr(runs,'load_checkpoint',bad)
    with pytest.raises(ValueError,match='counters'):runs.optimizer_preflight(config())


def test_interrupted_preflight_keeps_unknown_work(tmp_path,monkeypatch):
    saved=[]
    monkeypatch.setattr(runs,'balance_preflight',lambda c:(_ for _ in ()).throw(WallBudget('inside parent')))
    with pytest.raises(WallBudget):runs.optimizer_preflight(config(),save_partial=lambda d:saved.append(copy.deepcopy(d)))
    assert saved[-1]['compute']['operations'][0]['status']=='started'
    assert saved[-1]['compute']['operations'][0]['completed_frames'] is None
    assert saved[-1]['compute']['completed_frame_evaluations']=={}
