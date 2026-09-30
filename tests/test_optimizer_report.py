import importlib.util
import json
from pathlib import Path
import pytest


def generator():
    spec=importlib.util.spec_from_file_location('optimizer_report','scripts/report_optimizer_steps.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(root,result):
    root.mkdir();(root/'summary.json').write_text(json.dumps(result));(root/'config.json').write_text('{}')
    for name,case in result['cases'].items():
        (root/name).mkdir();(root/name/'summary.json').write_text(json.dumps(case))


@pytest.mark.parametrize('attempted',[False,True])
def test_preflight_and_partial_report_actual_counts(tmp_path,attempted):
    cases={n:{'status':'blocked','executed':False} for n in ('replay','balanced','force-only')}
    if attempted:cases['replay']={'status':'failed','executed':True,'failure_reason':'FD interrupted','counts':{'fd_cells':100,'fd_steps':999,'probes':666},'directions':{'stateful_adam':{'roles':{'fitting':{'finite_differences':{'objectives':{'energy':{'steps':[{'valid':False}]}}}}}}},'compute':{'operations':[{'status':'started','completed_frames':None}],'completed_frame_evaluations':{}}}
    root=tmp_path/'raw';write(root,{'status':'failed','execution_valid':False,'failure_reason':'fixture failure','cases':cases});out=tmp_path/'out';generator().generate(root,out)
    c=json.loads((out/'optimizer-step-diagnostics.json').read_text());md=(out/'optimizer-step-diagnostics.md').read_text()
    assert c['attempted_cases']==int(attempted) and c['blocked_cases']==3-int(attempted)
    assert c['actual_counts']=={'fd_cells':int(attempted),'fd_steps':int(attempted),'probes':0}
    assert 'Scientific interpretation blocked' in md and 'Full pilot: **pending**' in md
    assert not c['learning_accepted'] and not c['interpretation_valid']
    if attempted:assert 'FD interrupted' in md and c['unconfirmed_operations']==1


def valid_case():
    metrics={'fitting':{'energy_mse_eV2':.1,'force_mse_eV2_per_A2':.2},'anchor':{'energy_mse_eV2':.1,'force_mse_eV2_per_A2':.2}}
    roles={r:{'original_baseline_fraction':{'energy_mse_eV2':1.,'force_mse_eV2_per_A2':1.},'source_relative_change':{'energy_mse_eV2':0.,'force_mse_eV2_per_A2':0.}} for r in ('fitting','anchor')}
    na={'status':'not_applicable_zero_direction','valid':None,'objectives':{}}
    return {'status':'completed','executed':True,'verification_valid':True,'state_hashes_before':{'p':'x'},'state_hashes_after':{'p':'x'},
            'adam_identity':{'valid':True},'weighted_gradient_identity':{'valid':True},'builder_identity':{'valid':True},
            'counts':{'fd_cells':0,'fd_steps':0,'probes':2},'training_frame_exposures':0,
            'directions':{'stateful_adam':{'status':'zero_direction','norm':0.,'roles':{r:{'finite_differences':na} for r in ('fitting','anchor')}},
                          'reset_adam_matched_norm':{'status':'unavailable','norm':None,'reason':'zero_target_norm','roles':{}},
                          'steepest_matched_norm':{'status':'unavailable','norm':None,'reason':'zero_target_norm','roles':{}}},
            'probes':[{'direction':'stateful_adam','scale':s,'metrics':metrics,'roles':roles} for s in (.1,1.)]}


def test_unavailable_is_preserved_and_not_learning_acceptance(tmp_path):
    root=tmp_path/'raw';write(root,{'status':'completed','execution_valid':True,'source_hashes_unchanged':True,'learning_accepted':True,'cases':{n:valid_case() for n in ('replay','balanced','force-only')}})
    out=tmp_path/'out';generator().generate(root,out);c=json.loads((out/'optimizer-step-diagnostics.json').read_text());md=(out/'optimizer-step-diagnostics.md').read_text()
    assert c['interpretation_valid'] and not c['learning_accepted'] and c['actual_counts']['fd_cells']==0
    assert 'unavailable' in md and 'zero_target_norm' in md and 'fully fitted' in md and 'causal' in md


@pytest.mark.parametrize('failure',['fd','hash','builder'])
def test_report_gates_forged_success_with_failed_verification(tmp_path,failure):
    case=valid_case()
    if failure=='fd':case['directions']['stateful_adam']={'status':'available','roles':{'fitting':{'finite_differences':{'valid':False,'objectives':{'energy':{'valid':False,'steps':[{'valid':False}]}}}}}}
    if failure=='hash':case['state_hashes_after']={'p':'y'}
    if failure=='builder':case['builder_identity']['valid']=False
    root=tmp_path/'raw';write(root,{'status':'completed','execution_valid':True,'source_hashes_unchanged':True,'cases':{n:case for n in ('replay','balanced','force-only')}})
    out=tmp_path/'out';generator().generate(root,out);c=json.loads((out/'optimizer-step-diagnostics.json').read_text())
    assert not c['interpretation_valid'] and not c['learning_accepted']


@pytest.mark.parametrize('damage',['directions','stateful','role','objective','step','probe','scale','unavailable_reason'])
def test_incomplete_or_altered_matrix_cannot_be_interpreted(damage):
    import copy
    c=copy.deepcopy(json.loads(Path('runs/aspirin-optimizer-step-2026-09-30/replay/observations.json').read_text()))
    assert generator().verified(c)
    if damage=='directions':c['directions']={};c['probes']=[]
    if damage=='stateful':c['directions'].pop('stateful_adam')
    if damage=='role':c['directions']['stateful_adam']['roles'].pop('anchor')
    if damage=='objective':c['directions']['stateful_adam']['roles']['fitting']['finite_differences']['objectives'].pop('force')
    if damage=='step':c['directions']['stateful_adam']['roles']['fitting']['finite_differences']['objectives']['energy']['steps'].pop()
    if damage=='probe':c['probes'].pop()
    if damage=='scale':c['probes'][0]['scale']=.2
    if damage=='unavailable_reason':
        c['directions']['reset_adam_matched_norm']={'status':'unavailable','norm':None,'roles':{}}
        c['probes']=[p for p in c['probes'] if p['direction']!='reset_adam_matched_norm']
    c['counts']=generator().counts(c)
    assert not generator().verified(c)
