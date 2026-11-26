import copy
import json
from pathlib import Path
import pytest
from test_coverage import frozen_case


def test_replay_requires_every_scheduled_value_and_complete_verified_arm():
    from force_consistency.coverage_matrix import verify_replay
    initial={'energy_mse_eV2':.1,'force_mse_eV2_per_A2':.01};final={'energy_mse_eV2':.001,'force_mse_eV2_per_A2':.007}
    reference=[{'update':0,**initial},{'update':100,**final}]
    good={'execution_valid':True,'updates':100,'checkpoint_new_process_verified':True,
          'history':[{'update':r['update'],'metrics':{'fitting':{k:r[k] for k in initial}}} for r in reference]}
    assert verify_replay(good,reference,100,{'atol':1e-10,'rtol':1e-8})['status']=='passed'
    for changed in [{**good,'updates':99},{**good,'checkpoint_new_process_verified':False},{**good,'history':good['history'][:1]}]:
        with pytest.raises(ValueError): verify_replay(changed,reference,100,{'atol':1e-10,'rtol':1e-8})
    bad=copy.deepcopy(good);bad['history'][1]['metrics']['fitting']['force_mse_eV2_per_A2']*=1.01
    with pytest.raises(ValueError): verify_replay(bad,reference,100,{'atol':1e-10,'rtol':1e-8})


def test_matrix_learning_failure_allows_c_but_operational_failure_blocks(tmp_path):
    from force_consistency.coverage_matrix import execute_matrix
    def run_case(path,b_status):
        calls=[]
        def runner(name,expected):
            calls.append(name)
            result={'execution_valid':True,'status':'completed','updates':2,'checkpoint_new_process_verified':True,
                    'initial_state_hashes':{'params':'p','optimizer':'o'},'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
            if name=='diverse16': result.update(b_status)
            return result
        result=execute_matrix(path,runner,lambda a:{'status':'passed'})
        return calls,result
    calls,result=run_case(tmp_path/'finite',{})
    assert calls==['narrow16','diverse16','diverse128'] and result['execution_valid']
    assert result['full_pilot']=='pending'
    for i,status in enumerate([{'execution_valid':False,'status':'failed'},{'execution_valid':False,'status':'incomplete'}]):
        calls,result=run_case(tmp_path/f'bad{i}',status)
        assert calls==['narrow16','diverse16'] and result['arms']['diverse128']['status']=='blocked' and not result['execution_valid']
    calls=[]
    def failed_replay(a): raise ValueError('Mismatch')
    result=execute_matrix(tmp_path/'replay',lambda name,expected:calls.append(name) or {'execution_valid':True},failed_replay)
    assert calls==['narrow16'] and result['arms']['diverse16']['status']=='blocked'


def test_preflight_stale_reference_never_opens_arrays(tmp_path,monkeypatch):
    import force_consistency.coverage_matrix as cm
    p,m,n,planned=frozen_case()
    c=json.loads(Path('configs/aspirin-data-coverage-diagnostic.json').read_text())
    c['run_directory']=str(tmp_path/'stale');c.pop('evidence_log');c['reference_sha256']['history.json']='stale'
    monkeypatch.setattr(cm,'load_training_partition',lambda *a,**kw:pytest.fail('Arrays opened with stale reference'))
    result=cm.diagnose_coverage(c)
    assert result['status']=='failed' and not result['execution_valid']
    assert all(not arm['executed'] for arm in result['arms'].values())
    assert 'identity' in result['failure_reason']
    with pytest.raises(ValueError,match='not empty'): cm.diagnose_coverage(c)


def test_cli_matrix_with_real_train_only_preparation_and_fresh_probes(tmp_path):
    from dataclasses import asdict,replace
    import numpy as np
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.audit import Connection,ConnectionInventory
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.data.prepare import export_prepared,load_training_partition,write_json
    from force_consistency.data.splits import assign_groups,canonical_hash
    from force_consistency.coverage import select_coverage_subsets,take_subset
    from force_consistency.coverage_runs import run_coverage_arm,MSE_KEYS
    from force_consistency.diagnostics import sha256
    from force_consistency.cli import main
    rs,oldinv,ends=fixture();records=[r for r in rs if r.kind=='minimum'];connections={};newends={}
    for oldg,conn in oldinv.connections['aspirin'].items():
        g='01' if oldg=='010' else oldg;branches={}
        for b in ('minus','plus'):
            template=next(r for r in rs if r.connection_id==oldg and r.branch==b)
            count=(123 if b=='minus' else 111) if g=='01' else 16
            branch=[]
            for i in range(count):
                rid=f'Landscape17/aspirin/SteepestDescent/{g}/coords{b}_{i}.xyz';branch.append(rid)
                records.append(replace(template,record_id=rid,source_path=rid,connection_id=g,frame_index=i,
                    positions=np.array([[0.,0.,0.],[1.+i*.001,.01,0.]]),energy=i*.002,forces=np.full((2,3),.01+i*.00001)))
            branches[b]=tuple(branch)
            newends[f'aspirin/{g}/{b}']={**ends[f'aspirin/{oldg}/{b}']}
        template=next(r for r in rs if r.connection_id==oldg and r.kind=='transition_state')
        ts=f'aspirin/{g}/ts';records.append(replace(template,record_id=ts,connection_id=g))
        connections[g]=Connection('aspirin',g,branches,ts,('0','0'))
    inv=ConnectionInventory({'aspirin':connections},oldinv.minima,{})
    policy=next(PreparationPolicy(seed=seed) for seed in range(20) if '01' in assign_groups(inv,PreparationPolicy(seed=seed)).groups['aspirin']['train'])
    data=tmp_path/'data';m=export_prepared(records,directory=data,inventory=inv,endpoints=newends,duplicates=empty_duplicates(),policy=policy,molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    for name in ('validation','test','validation_minima_diagnostics'): (data/f'{name}.npz').unlink()
    p,_,norm=load_training_partition(data,expected_split_hash=m['split_hash'])
    ids=[f'Landscape17/aspirin/SteepestDescent/01/coords{b}_{r}.xyz' for b,ranks in [('minus',[0,15,31,46,62,77,93,108]),('plus',[1,16,32,47,63,78,94,110])] for r in ranks]
    selected=select_coverage_subsets(p,m,ids);anchor_file=tmp_path/'anchor.json'
    write_json(anchor_file,selected['subsets']['narrow16'])
    planned={**selected,'schema':1,'status':'planned_unexecuted','anchor_source':str(anchor_file),'anchor_source_sha256':sha256(anchor_file)}
    planned['identity']=canonical_hash(planned);manifest_file=tmp_path/'subsets.json';write_json(manifest_file,planned)
    from test_coverage import arm_config
    c={**arm_config(norm),'microbatch_size':16,'max_updates':1,'data_directory':str(data),'expected_split_hash':m['split_hash'],
       'expected_prepared_identity':m['identity'],'subsets_manifest':str(manifest_file),'expected_subsets_identity':planned['identity'],
       'run_directory':str(tmp_path/'matrix'),'reference_run':str(tmp_path/'reference'),'replay_tolerance':{'atol':1e-10,'rtol':1e-8}}
    # Engineering fixture reference, not historical research evidence.
    ref=run_coverage_arm(c,take_subset(p,ids),take_subset(p,ids),tmp_path/'reference',{})
    assert ref['execution_valid']
    write_json(tmp_path/'reference/history.json',[{'update':row['update'],**{k:row['metrics']['fitting'][k] for k in MSE_KEYS}} for row in ref['history']])
    write_json(tmp_path/'reference/config.json',{**c,'batch_size':16})
    c['reference_sha256']={n:sha256(tmp_path/'reference'/n) for n in ('config.json','provenance.json','summary.json','history.json','checkpoint-verification.json')}
    config_file=tmp_path/'config.json';write_json(config_file,c)
    assert main(['diagnose-coverage','--config',str(config_file)])==0
    result=json.loads((tmp_path/'matrix/summary.json').read_text())
    assert result['execution_valid'] and all(a['checkpoint_new_process_verified'] for a in result['arms'].values())
    assert [a['frame_exposures'] for a in result['arms'].values()]==[128,16,16]  # sorted JSON arm keys
    # Prepared-array corruption fails before any arm is executed.
    train=data/'train.npz'
    with np.load(train) as z: arrays={k:z[k] for k in z.files}
    arrays['forces']=arrays['forces']*2;np.savez(train,**arrays)
    c['run_directory']=str(tmp_path/'corrupt');write_json(config_file,c)
    assert main(['diagnose-coverage','--config',str(config_file)])==1
    failed=json.loads((tmp_path/'corrupt/summary.json').read_text())
    assert all(not a['executed'] for a in failed['arms'].values()) and 'identity' in failed['failure_reason'].lower()


@pytest.mark.parametrize('saved_result',[False,True])
def test_runner_exception_retains_partial_or_completed_artifacts(tmp_path,saved_result):
    from force_consistency.coverage_matrix import execute_matrix
    from force_consistency.data.prepare import write_json
    root=tmp_path/'matrix'
    original={'status':'completed','execution_valid':True,'executed':True,'updates':3,'history':[{'update':3}],
              'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    def runner(name,expected):
        path=root/name;path.mkdir(parents=True)
        write_json(path/'history.json',[{'update':3}])
        (path/'last.msgpack').write_bytes(b'preserved witness checkpoint')
        if saved_result: write_json(path/'summary.json',original)
        raise OSError('Original interrupted output failure')
    result=execute_matrix(root,runner,lambda a:{'status':'passed'})
    assert result['status']=='failed' and 'Original interrupted output failure' in result['failure_reason']
    arm=result['arms']['narrow16']
    assert arm['status']=='failed' and arm['executed'] and arm['updates']==3
    assert all(result['arms'][n]['status']=='blocked' for n in ('diverse16','diverse128'))
    assert (root/'narrow16/last.msgpack').read_bytes()==b'preserved witness checkpoint'
    assert json.loads((root/'narrow16/history.json').read_text())==[{'update':3}]
    if saved_result: assert json.loads((root/'narrow16/summary.json').read_text())==original
    assert json.loads((root/'summary.json').read_text())['status']=='failed'


def test_outer_failure_preserves_completed_arm_and_blocks_only_unexecuted(tmp_path,monkeypatch):
    import force_consistency.coverage_matrix as cm
    from force_consistency.data.prepare import write_json
    c=json.loads(Path('configs/aspirin-data-coverage-diagnostic.json').read_text());c.pop('evidence_log')
    root=tmp_path/'outer';c['run_directory']=str(root)
    original={'status':'completed','execution_valid':True,'executed':True,'updates':2000}
    def unexpected(root,runner,replay):
        path=Path(root)/'narrow16';path.mkdir();write_json(path/'summary.json',original)
        raise OSError('Interrupted matrix bookkeeping')
    monkeypatch.setattr(cm,'execute_matrix',unexpected)
    result=cm.diagnose_coverage(c)
    assert result['status']=='failed' and 'Interrupted matrix bookkeeping' in result['failure_reason']
    assert result['arms']['narrow16']==original
    assert json.loads((root/'narrow16/summary.json').read_text())==original
    assert all(result['arms'][name]['status']=='blocked' for name in ('diverse16','diverse128'))
