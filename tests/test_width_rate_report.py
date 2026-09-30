"""Saved-record report guards. Fixtures are synthetic metadata, never research results."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from force_consistency.width_rate_contract import verify_width_rate_sources,canonical_hash,digest,MSE_KEYS
from test_width_rate_preflight import config


def generator():
    spec=importlib.util.spec_from_file_location('width_rate_report','scripts/report_width_rate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def gate(metrics,base):
    ratios={k:metrics[k]/base[k] for k in MSE_KEYS}
    return {'accepted':all(v<.1 for v in ratios.values()),'fraction_remaining':ratios,'threshold_mse':{k:base[k]*.1 for k in MSE_KEYS},'gate_fraction':.1}


def valid_arm(case=None):
    c=config();sources=verify_width_rate_sources(c);case=c['cases'][0] if case is None else case
    norm=json.loads(Path('runs/aspirin-loss-balance-2026-09-30/balanced/config.json').read_text())['normalization']
    initial={'params':('a' if case['features']==16 else 'c')*64,'optimizer':('b' if case['features']==16 else 'd')*64}
    final={'params':'e'*64,'optimizer':'f'*64};base=copy.deepcopy(sources['original_baselines']);h=[]
    for u in c['score_updates']:
        # Historical curves provide realistic complete metric topology; update1 is
        # intentionally synthetic in this engineering-only saved-record fixture.
        metrics=copy.deepcopy(sources['historical'].get(u,base));g={r:gate(metrics[r],base[r]) for r in ('fitting','anchor')}
        h.append({'update':u,'frame_exposures':u*128,'metrics':metrics,'fit_gate':g['fitting'],'anchor_gate':g['anchor'],
          'original_baseline_fraction':{r:g[r]['fraction_remaining'] for r in g},
          'weighted_loss':{r:c['loss_weights']['energy']*metrics[r][MSE_KEYS[0]]/norm['energy_loss_scale_eV']**2+c['loss_weights']['force']*metrics[r][MSE_KEYS[1]]/norm['force_loss_scale_eV_per_A']**2 for r in g}})
    v={'status':'passed','update':6000,'adam_count':6000,'normalization':norm,'state_hashes':final,'checkpoint_sha256':'1'*64,'history_sha256':'2'*64,
       'case':case,'config_identity':canonical_hash(c),'fitting_record_count':128,'anchor_record_count':16,'maximum_absolute_error':0.,'tolerance':1e-10,
       'precision':'float64','template_initializations':1,'template_initializer_frames':16,'prediction_frames':144}
    oracle={'status':'passed','count':1,'discarded':True,'atol':1e-10,'rtol':1e-8,'state_maximum_absolute_difference':0.,'prediction_maximum_absolute_difference':0.,
            'state_maximum_tolerance_fraction':0.,'prediction_maximum_tolerance_fraction':0.,'fullbatch_reference_frames':128,'accumulated_proposal_frames':128,'prediction_reference_frames':256,'initial_state_hashes':initial,'proposed_state_hashes':final,'gradient_sha256':'3'*64,'elapsed_seconds':10.}
    a={'case':case,'status':'completed','executed':True,'execution_valid':True,'full_pilot':'pending','stop_reason':'fixed_budget','source_hashes_unchanged':True,
       'updates':6000,'attempted_updates':6000,'persisted_update':6000,'pending_update_unknown':False,'frame_exposures':768000,'persisted_frame_exposures':768000,
       'warmup_frame_exposures':128,'warmup_source_unchanged':True,'warmup_work_status':'completed','oracle_work_status':'completed','oracle_verified':True,
       'oracle_verification':oracle,'oracle_metrics':h[1]['metrics'],'oracle_metrics_prediction_frames':144,'checkpoint_new_process_verified':True,
       'checkpoint_verification':v,'checkpoint_sha256':'1'*64,'initial_state_hashes':initial,'initial_state_hashes_after':initial,
       'initial_raw_state_hashes':sources['historical_initial_state_hashes'] if case['features']==16 else initial,
       'final_state_hashes':final,'history':h,'final_metrics':h[-1]['metrics'],'fit_gate':h[-1]['fit_gate'],'anchor_gate':h[-1]['anchor_gate'],'original_baselines':base,
       'fitting_frame_count':128,'anchor_frame_count':16,'evaluation_passes':64,'evaluation_frame_count':9216,'reference_prediction_frames':144,
       'replay_verified':True,'replay_compared_updates':list(sources['historical']) if case['name']=='width16-lr1e-3' else [],'replay_maximum_absolute_mse_difference':0.,
       'elapsed_seconds':1000.,'shared_initialization_seconds':2.,'charged_elapsed_seconds':1002.,'parameter_count':100 if case['features']==16 else 200,
       'compile_seconds':1.,'warmup_seconds':1.,'first_step_reference_seconds':10.,'probe_seconds':1.,'checkpoint_seconds':1.,'steady_update_seconds_total':900.,
       'steady_update_seconds_mean':.15,'evaluation_seconds_total':50.,
       'provenance':{'config_identity':canonical_hash(c),'case':case,'normalization':norm,'original_baselines':base,'initial_state_hashes':initial,
           'inputs':sources['historical_inputs'],'precision':'float64','backend':'cpu','model':{**c['model'],'features':case['features']},
           'source_pin_manifest_identity':c['source_pin_manifest_sha256'],'sources':json.loads(json.dumps(sources))}}
    return c,sources,json.loads(json.dumps(a))


def test_valid_complete_failure_is_reportable_without_pilot_clearance():
    c,s,a=valid_arm();m=generator();assert m.verified_arm(a,c,s)
    assert not a['fit_gate']['accepted'] and a['full_pilot']=='pending'

@pytest.mark.parametrize('damage',['schedule','duplicate','baseline','source1','pins','receipt','features','rate','support','updates','persisted','attempted','cap','charged','norm','warmup','oracle','oracle_count','state','precision','rawgate','weighted','pergroup','equalgroup','overlap'])
def test_edited_or_incomplete_saved_claim_is_rejected(damage):
    c,s,a=valid_arm();m=generator()
    if damage=='schedule':a['history'].pop(4)
    elif damage=='duplicate':a['history'][3]['update']=1
    elif damage=='baseline':
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:a['original_baselines'][role][key]*=1000
        for row in a['history']:
            for role,label in [('fitting','fit_gate'),('anchor','anchor_gate')]:
                row[label]=gate(row['metrics'][role],a['original_baselines'][role]);row['original_baseline_fraction'][role]=row[label]['fraction_remaining']
    elif damage=='source1':a['replay_compared_updates'].append(1)
    elif damage=='pins':a['provenance']['sources']['pins'].pop(next(iter(a['provenance']['sources']['pins'])))
    elif damage=='receipt':a.pop('checkpoint_verification')
    elif damage=='features':a['case']={**a['case'],'features':32}
    elif damage=='rate':a['case']={**a['case'],'learning_rate':.002}
    elif damage=='support':a['fitting_frame_count']=127
    elif damage=='updates':a['updates']=6010
    elif damage=='persisted':a['persisted_update']=5900
    elif damage=='attempted':a['attempted_updates']=6001
    elif damage=='cap':a['elapsed_seconds']=1801
    elif damage=='charged':a['shared_initialization_seconds']=1000
    elif damage=='norm':a['provenance']['normalization']['energy_offset_eV']+=1
    elif damage=='warmup':a['warmup_frame_exposures']=256
    elif damage=='oracle':a['oracle_metrics']['fitting'][MSE_KEYS[0]]*=2
    elif damage=='oracle_count':a['oracle_verification']['fullbatch_reference_frames']=127
    elif damage=='state':a['initial_state_hashes_after']['params']='0'*64
    elif damage=='precision':a['checkpoint_verification']['precision']='float32'
    elif damage=='rawgate':a['history'][2]['fit_gate']['accepted']=True
    elif damage=='weighted':a['history'][2]['weighted_loss']['fitting']*=2
    elif damage=='pergroup':a['history'][2]['metrics']['fitting']['per_group'].pop(next(iter(a['history'][2]['metrics']['fitting']['per_group'])))
    elif damage=='equalgroup':a['history'][2]['metrics']['fitting']['equal_group'][MSE_KEYS[1]]*=2
    else:a['history'][2]['metrics']['anchor_fitted_flags'][next(iter(a['history'][2]['metrics']['anchor_fitted_flags']))]=False
    assert not m.verified_arm(a,c,s)


def test_four_complete_cells_only_and_positive_denominators():
    m=generator();c,s,a=valid_arm();arms={case['name']:{**valid_arm(case)[2],'interpretation_valid':True} for case in c['cases']}
    contrasts=m.factorial_contrasts(arms)
    assert contrasts['available'] and contrasts['ratios']['fitting'][MSE_KEYS[1]]['rate_within_width16']==1.
    assert contrasts['ratios']['fitting'][MSE_KEYS[1]]['interaction_ratio_of_rate_ratios']==1.
    for kind in ('missing','partial','zero','nonfinite'):
        bad=copy.deepcopy(arms);name=c['cases'][0]['name']
        if kind=='missing':bad.pop(name)
        elif kind=='partial':bad[name]['execution_valid']=False
        else:bad[name]['final_metrics']['fitting'][MSE_KEYS[1]]=0. if kind=='zero' else float('nan')
        assert not m.factorial_contrasts(bad)['available']


def saved_fixture(root):
    c,s,a=valid_arm();root.mkdir();(root/'config.json').write_text(json.dumps(c));(root/'sources.json').write_text(json.dumps(s));arms={}
    for case in c['cases']:
        _,_,arm=valid_arm(case);child=root/case['name'];child.mkdir();(child/'snapshots').mkdir();(child/'histories').mkdir()
        # Deliberately opaque engineering bytes: report verifies hashes and receipt
        # bindings; physical restore/predictions are separate runner/probe tests.
        (child/'snapshots/6000.msgpack').write_bytes(b'engineering-only-opaque-checkpoint')
        (child/'histories/6000.json').write_text(json.dumps(arm['history']))
        commit={'update':6000,'checkpoint':'snapshots/6000.msgpack','history':'histories/6000.json',
                'checkpoint_sha256':digest(child/'snapshots/6000.msgpack'),'history_sha256':digest(child/'histories/6000.json'),'state_hashes':arm['final_state_hashes']}
        (child/'commit.json').write_text(json.dumps(commit))
        for role in ('fitting','anchor'):(child/f'{role}-inputs.npz').write_bytes(b'engineering-only-opaque-inputs-'+role.encode())
        (child/'reference.npz').write_bytes(b'engineering-only-opaque-predictions')
        arm['provenance']['input_files_sha256']={r:digest(child/f'{r}-inputs.npz') for r in ('fitting','anchor')}
        contract={'case':case,'config_identity':canonical_hash(c),'update':6000,'normalization':s['normalization'],'commit':commit,
                  'state_hashes':arm['final_state_hashes'],'inputs':s['historical_inputs'],'input_files_sha256':arm['provenance']['input_files_sha256'],'reference_sha256':digest(child/'reference.npz')}
        (child/'checkpoint-contract.json').write_text(json.dumps(contract))
        arm['checkpoint_sha256']=commit['checkpoint_sha256'];arm['checkpoint_verification'].update(checkpoint_sha256=commit['checkpoint_sha256'],history_sha256=commit['history_sha256'])
        for file,data in [('config.json',c),('case.json',case),('provenance.json',arm['provenance']),('oracle-verification.json',{**arm['oracle_verification'],'metrics':arm['oracle_metrics']}),('checkpoint-verification.json',arm['checkpoint_verification']),('summary.json',arm)]:
            (child/file).write_text(json.dumps(data))
        arms[case['name']]=arm
    summary={'status':'completed','execution_valid':True,'full_pilot':'pending','elapsed_seconds':4010.,'arms':arms,
             'parent_work':{'status':'completed','historical_template_initializations':1,'completed_source_prediction_frames':208,'elapsed_seconds':2.},
             'shared_initializations':{str(width):{'features':width,'status':'completed','completed_initializations':1,'elapsed_seconds':2.,
                 'baseline_prediction_frames':144,'initial_metrics':s['original_baselines'],'initial_state_hashes':arms[f'width{width}-lr1e-3']['initial_state_hashes'],'parameter_count':100 if width==16 else 200} for width in (16,32)}}
    parent={**summary['parent_work'],'operation':'historical_parent_preflight'}
    shared={k:{**v,'operation':'shared_training_initialization'} for k,v in summary['shared_initializations'].items()}
    summary['parent_work']=parent;summary['shared_initializations']=shared
    (root/'preflight-work.json').write_text(json.dumps([parent,*shared.values()]))
    (root/'summary.json').write_text(json.dumps(summary));return c,s,summary

@pytest.mark.parametrize('damage',['history_pair','copied_summary','swapped_initial','global_cap'])
def test_generate_reconciles_saved_pairs_and_matrix_ancestry(tmp_path,damage):
    c,s,summary=saved_fixture(tmp_path/'runs');root=tmp_path/'runs';m=generator()
    if damage=='history_pair':(root/c['cases'][0]['name']/'histories/6000.json').write_text('[]')
    elif damage=='copied_summary':summary['arms'][c['cases'][0]['name']]['updates']=6010;(root/'summary.json').write_text(json.dumps(summary))
    elif damage=='swapped_initial':
        name=c['cases'][1]['name'];arm=summary['arms'][name];arm['initial_state_hashes']['params']='9'*64
        (root/name/'summary.json').write_text(json.dumps(arm));(root/'summary.json').write_text(json.dumps(summary))
    else:summary['elapsed_seconds']=11401.;(root/'summary.json').write_text(json.dumps(summary))
    out=m.generate(root,tmp_path/'out');assert not out['interpretation_valid'] and not out['contrasts']['available']
    assert out['full_pilot']=='pending'


def test_report_runs_without_importing_jax_or_loading_arrays(tmp_path):
    c,s,summary=saved_fixture(tmp_path/'runs')
    code='''import importlib.abc,sys,runpy
class RejectNumerics(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('jax','flax','optax','e3x'):raise AssertionError('Numerical import in reporter: '+name)
sys.meta_path.insert(0,RejectNumerics())
sys.argv=['scripts/report_width_rate.py','--root',sys.argv[1],'--output',sys.argv[2]]
runpy.run_path('scripts/report_width_rate.py',run_name='__main__')
'''
    proc=subprocess.run([sys.executable,'-c',code,str(tmp_path/'runs'),str(tmp_path/'out')],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    out=json.loads((tmp_path/'out/width-rate-diagnostics.json').read_text());assert out['interpretation_valid'] and out['contrasts']['available']
    assert out['actual_counts']['synchronized_updates']==24000 and out['actual_counts']['committed_frame_exposures']==3072000
    assert out['full_pilot']=='pending'


def test_oracle_pass_claim_with_excessive_normalized_discrepancy_is_rejected():
    c,s,a=valid_arm();a['oracle_verification']['state_maximum_tolerance_fraction']=2.
    assert not generator().verified_arm(a,c,s)


def test_impossible_global_elapsed_is_rejected_despite_under_cap(tmp_path):
    c,s,summary=saved_fixture(tmp_path/'runs');summary['elapsed_seconds']=0.
    (tmp_path/'runs/summary.json').write_text(json.dumps(summary))
    out=generator().generate(tmp_path/'runs',tmp_path/'out')
    assert not out['interpretation_valid'] and not out['contrasts']['available']


def test_negative_parent_elapsed_cannot_reconcile_global_time(tmp_path):
    c,s,summary=saved_fixture(tmp_path/'runs');summary['parent_work']['elapsed_seconds']=-20.
    (tmp_path/'runs/summary.json').write_text(json.dumps(summary))
    # Coherently edit the raw work ledger too; finite nonnegative durations are a constraint.
    events=json.loads((tmp_path/'runs/preflight-work.json').read_text());events[0]['elapsed_seconds']=-20.
    (tmp_path/'runs/preflight-work.json').write_text(json.dumps(events))
    out=generator().generate(tmp_path/'runs',tmp_path/'out')
    assert not out['interpretation_valid'] and not out['contrasts']['available']


def test_interrupted_preflight_counts_completed_width_once_and_marks_unknown(tmp_path):
    c=config();s=verify_width_rate_sources(c);root=tmp_path/'runs';root.mkdir()
    (root/'config.json').write_text(json.dumps(c));(root/'sources.json').write_text(json.dumps(s))
    events=[{'operation':'historical_parent_preflight','status':'started','historical_template_initializations':None,'completed_source_prediction_frames':None,'elapsed_seconds':0.},
      {'operation':'historical_parent_preflight','status':'completed','historical_template_initializations':1,'completed_source_prediction_frames':208,'elapsed_seconds':1.},
      {'operation':'shared_training_initialization','features':16,'status':'started','completed_initializations':None},
      {'operation':'shared_training_initialization','features':16,'status':'completed','completed_initializations':1,'elapsed_seconds':2.},
      {'operation':'shared_training_initialization','features':32,'status':'started','completed_initializations':None}]
    summary={'status':'failed','execution_valid':False,'full_pilot':'pending','elapsed_seconds':5.,'preflight_work':events,
             'arms':{case['name']:{'case':case,'status':'blocked','executed':False,'execution_valid':False} for case in c['cases']}}
    (root/'summary.json').write_text(json.dumps(summary));(root/'preflight-work.json').write_text(json.dumps(events))
    out=generator().generate(root,tmp_path/'out');counts=out['actual_counts']
    assert counts['shared_training_initializations_completed']==1
    assert counts['parent_work']['historical_template_initializations']==1
    assert counts['parent_work']['completed_source_prediction_frames']==208
    assert counts['unknown_pending_preflight_operations']==['shared_training_initialization:width32']
    assert not counts['work_counts_complete'] and counts['synchronized_updates']==0


def test_coherently_rehashed_nonreplay_initial_metrics_must_match_pinned_original(tmp_path):
    c,s,summary=saved_fixture(tmp_path/'runs');root=tmp_path/'runs';name=c['cases'][1]['name'];arm=summary['arms'][name];child=root/name
    arm['history'][0]=copy.deepcopy(arm['history'][2]);arm['history'][0].update(update=0,frame_exposures=0)
    (child/'histories/6000.json').write_text(json.dumps(arm['history']))
    receipt=json.loads((child/'commit.json').read_text());receipt['history_sha256']=digest(child/'histories/6000.json')
    (child/'commit.json').write_text(json.dumps(receipt));arm['checkpoint_verification']['history_sha256']=receipt['history_sha256']
    contract=json.loads((child/'checkpoint-contract.json').read_text());contract['commit']=receipt;(child/'checkpoint-contract.json').write_text(json.dumps(contract))
    (child/'checkpoint-verification.json').write_text(json.dumps(arm['checkpoint_verification']));(child/'summary.json').write_text(json.dumps(arm));(root/'summary.json').write_text(json.dumps(summary))
    out=generator().generate(root,tmp_path/'out')
    assert not out['interpretation_valid'] and not out['contrasts']['available']


def test_legacy_failed_probe_retains_bound_reference_work_and_unknown_duration(tmp_path):
    c,s,summary=saved_fixture(tmp_path/'runs');root=tmp_path/'runs';name=c['cases'][0]['name'];arm=summary['arms'][name];child=root/name
    arm.update(status='failed',execution_valid=False,checkpoint_new_process_verified=False,probe_stderr='ValueError: Checkpoint provenance mismatch',probe_seconds=0.)
    arm.pop('reference_prediction_frames');arm.pop('checkpoint_verification')
    (child/'checkpoint-verification.json').unlink()
    (child/'summary.json').write_text(json.dumps(arm));summary.update(status='failed',execution_valid=False)
    (root/'summary.json').write_text(json.dumps(summary))
    out=generator().generate(root,tmp_path/'out');counts=out['actual_counts']
    assert counts['unknown_probe_cells']==[name] and not counts['work_counts_complete']
    assert out['arms'][name]['derived_reference_prediction_frames']==144
    assert out['arms'][name]['derived_probe_seconds'] is None
    assert out['arms'][name]['probe_seconds']==0. # original failed receipt preserved
