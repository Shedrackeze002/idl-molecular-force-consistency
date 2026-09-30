import importlib.util
import json
from pathlib import Path
import pytest


def generator():
    spec=importlib.util.spec_from_file_location('loss_balance_report','scripts/report_loss_balance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(root,result):
    root.mkdir();(root/'summary.json').write_text(json.dumps(result));(root/'config.json').write_text('{}')
    for name,arm in result['arms'].items():
        (root/name).mkdir();(root/name/'summary.json').write_text(json.dumps(arm))


@pytest.mark.parametrize('attempted',[False,True])
def test_partial_and_preflight_reports_preserve_actual_counts(tmp_path,attempted):
    arms={name:{'status':'blocked','executed':False,'updates':0} for name in ('replay','balanced','force-only')}
    if attempted:arms['replay']={'status':'failed','executed':True,'updates':17,'failure_reason':'wall cap','history':[]}
    root=tmp_path/'raw';save(root,{'status':'failed','execution_valid':False,'arms':arms,'failure_reason':'fixture failure','next_step':'resolve_execution_failure'})
    out=tmp_path/'reports';generator().generate(root,out)
    evidence=json.loads((out/'loss-balance-diagnostics.json').read_text());report=(out/'loss-balance-diagnostics.md').read_text()
    assert evidence['attempted_arms']==int(attempted) and evidence['completed_arms']==0 and evidence['blocked_arms']==3-int(attempted)
    assert 'Scientific interpretation blocked' in report and 'fixture failure' in report
    if attempted:assert 'wall cap' in report and '17' in report
    assert 'Balanced joint acceptance: **False**' in report


def test_force_only_success_never_promoted_to_joint_or_generalization(tmp_path):
    arms={name:{'status':'completed','executed':True,'execution_valid':True,'updates':2000,
        'fit_gate':{'accepted':False,'fraction_remaining':{'energy_mse_eV2':.8,'force_mse_eV2_per_A2':.05}},
        'anchor_gate':{'accepted':False,'fraction_remaining':{'energy_mse_eV2':.8,'force_mse_eV2_per_A2':.05}},
        'history':[]} for name in ('replay','balanced','force-only')}
    root=tmp_path/'raw';save(root,{'status':'completed','execution_valid':True,'balanced_joint_accepted':False,'force_only_force_accepted':True,'arms':arms,'next_step':'review_fixed_loss_outcomes_before_new_intervention'})
    out=tmp_path/'reports';generator().generate(root,out);report=(out/'loss-balance-diagnostics.md').read_text()
    assert 'Balanced joint acceptance: **False**' in report and 'Force-only force acceptance: **True**' in report
    assert 'absolute energy gauge' in report and 'No causal attribution' in report and 'all 16 anchor records are fitted' in report
    assert 'Full pilot: **pending**' in report
