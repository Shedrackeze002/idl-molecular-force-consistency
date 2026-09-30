import copy
import importlib.util
import json
from pathlib import Path
import pytest


def generator():
    spec=importlib.util.spec_from_file_location('convergence_report','scripts/report_convergence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def valid_arm():
    c={'start_update':2000,'end_update':6000,'additional_updates':4000,'validation_interval':100,'gate_fraction':.1,
       'normalization':{'energy_loss_scale_eV':2.,'force_loss_scale_eV_per_A':1.,'energy_offset_eV':0.},'loss_weights':{'energy':1.,'force':1.},'continuation_wall_seconds':1200}
    keys=('energy_mse_eV2','force_mse_eV2_per_A2');baseline={r:dict(zip(keys,(4.,1.))) for r in ('fitting','anchor')}
    metrics={r:dict(zip(keys,(.8,.4))) for r in baseline};g={'accepted':False,'fraction_remaining':dict(zip(keys,(.2,.4))),'gate_fraction':.1,'threshold_mse':dict(zip(keys,(.4,.1)))}
    history=[]
    for u in [2000,2001,*range(2100,6001,100)]:
        history.append({'update':u,'additional_updates':u-2000,'additional_frame_exposures':(u-2000)*128,'frame_exposures':u*128,'metrics':metrics,
                        'fit_gate':g,'anchor_gate':g,'original_baseline_fraction':{r:g['fraction_remaining'] for r in baseline},'weighted_loss':{r:.6 for r in baseline}})
    hash={'params':'a'*64,'optimizer':'b'*64};v={'status':'passed','update':6000,'adam_count':6000,'normalization':c['normalization'],'state_hashes':hash,'checkpoint_sha256':'c'*64,'maximum_absolute_error':0.,'tolerance':1e-10,'fitting_record_count':128,'anchor_record_count':16}
    a={'status':'completed','executed':True,'execution_valid':True,'source_hashes_unchanged':True,'source_state_hashes_before':hash,'source_state_hashes_after':hash,
       'updates':6000,'additional_updates':4000,'attempted_updates':4000,'pending_update_unknown':False,'additional_frame_exposures':512000,'frame_exposures':768000,
       'fitting_frame_count':128,'anchor_frame_count':16,'evaluation_passes':42,'evaluation_frame_count':6048,'history':history,'warmup_frame_exposures':128,'warmup_source_unchanged':True,
       'oracle_verified':True,'oracle_metrics':metrics,'original_baselines':baseline,'checkpoint_new_process_verified':True,'checkpoint_verification':v,'checkpoint_sha256':'c'*64,
       'final_state_hashes':hash,'persisted_update':6000,'final_metrics':metrics,'fit_gate':g,'anchor_gate':g,'stop_reason':'fixed_budget','elapsed_seconds':700.,'provenance':{'normalization':c['normalization']}}
    return c,json.loads(json.dumps(a))


def test_exact_completed_record_and_no_rebased_baseline():
    c,a=valid_arm();m=generator();assert m.verified_arm(a,c)
    a['original_baselines']['fitting']['force_mse_eV2_per_A2']=.4
    assert not m.verified_arm(a,c)

@pytest.mark.parametrize('damage',['schedule','duplicate','count','additional','attempted','globalexposure','localexposure','oracle','receipt','source','warmup','rawgate','nonfinite','norm','persisted','elapsed'])
def test_incomplete_or_forged_completion_cannot_pass(damage):
    c,a=valid_arm()
    if damage=='schedule':a['history'].pop(5)
    elif damage=='duplicate':a['history'][2]['update']=2001
    elif damage=='count':a['updates']=5999
    elif damage=='additional':a['additional_updates']=4001
    elif damage=='attempted':a['pending_update_unknown']=True
    elif damage=='globalexposure':a['frame_exposures']=512000
    elif damage=='localexposure':a['additional_frame_exposures']=768000
    elif damage=='oracle':a['oracle_verified']=False
    elif damage=='receipt':a['checkpoint_verification']['adam_count']=5999
    elif damage=='source':a['source_state_hashes_after']['params']='d'*64
    elif damage=='warmup':a['warmup_frame_exposures']=0
    elif damage=='rawgate':a['fit_gate']['accepted']=True
    elif damage=='nonfinite':a['history'][-1]['metrics']['fitting']['force_mse_eV2_per_A2']=float('nan')
    elif damage=='norm':a['checkpoint_verification']['normalization']={}
    elif damage=='persisted':a['persisted_update']=5500
    elif damage=='elapsed':a['elapsed_seconds']=1201
    assert not generator().verified_arm(a,c)


def test_terminal_window_signed_math_zero_missing_and_no_theorem():
    m=generator();c,a=valid_arm();h=copy.deepcopy(a['history'])
    for row in h:
        if row['update']==5500:row['metrics']['fitting']['energy_mse_eV2']=2.
        if row['update']==6000:row['metrics']['fitting']['energy_mse_eV2']=2.01
    v=m.terminal_window(h,500)['fitting']['energy_mse_eV2'];assert v['relative_reduction']==pytest.approx(-.005) and v['label']=='small terminal-window change'
    assert m.terminal_window(h[:-1],500,end_update=6000)['fitting']['energy_mse_eV2']['label']=='unavailable'
    for row in h:
        if row['update']==5500:row['metrics']['fitting']['energy_mse_eV2']=0.
    assert m.terminal_window(h,500)['fitting']['energy_mse_eV2']['relative_reduction'] is None


def test_partial_preflight_report_preserves_zero_counts(tmp_path):
    m=generator();root=tmp_path/'run';root.mkdir();c,a=valid_arm();c.update(terminal_window_updates=500,small_window_relative_change=.01)
    s={'status':'failed','execution_valid':False,'full_pilot':'pending','arms':{n:{'status':'blocked','executed':False,'execution_valid':False,'additional_updates':0,'additional_frame_exposures':0} for n in ('replay','balanced','force-only')}}
    (root/'config.json').write_text(json.dumps(c));(root/'summary.json').write_text(json.dumps(s))
    result=m.generate(root,tmp_path/'reports')
    assert not result['interpretation_valid'] and not result['learning_accepted'] and result['actual_additional_frame_exposures']==0 and result['full_pilot']=='pending'
    assert result['arms']['force-only']['frame_exposures'] is None


def test_actual_historical_pins_are_verified_without_loading_arrays():
    c=json.loads(Path('configs/aspirin-convergence-budget.json').read_text());sources=json.loads(Path('runs/aspirin-optimizer-step-2026-09-30/sources.json').read_text())
    m=generator();assert m.verify_historical_pins(c,sources)
    c['calibration_sha256']='0'*64
    assert not m.verify_historical_pins(c,sources)
