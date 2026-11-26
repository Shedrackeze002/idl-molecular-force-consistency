"""Reporting regressions use altered temporary artifacts, never new research trials."""
import importlib.util
import json
from pathlib import Path
import shutil

spec=importlib.util.spec_from_file_location('gradient_report',Path('scripts/report_representation_gradients.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
SOURCE=Path('runs/aspirin-representation-gradient-2026-09-30')


def write(path,value):path.write_text(json.dumps(value)+'\n')


def test_failed_final_objective_is_not_reported_as_verified(tmp_path):
    root=tmp_path/'fixture';shutil.copytree(SOURCE,root)
    path=root/'final-C-anchor'/'observations.json';d=json.loads(path.read_text())
    d['verification_valid']=False;d['finite_differences']['energy']['valid']=False
    for step in d['finite_differences']['energy']['steps']:step['valid']=False
    d['finite_differences']['valid']=False;write(path,d)
    s=json.loads((root/'summary.json').read_text());s['execution_valid']=False;s['status']='failed'
    s['cases']['final-C-anchor'].update(status='failed',verification_valid=False)
    s['next_step']='resolve_verification_or_operational_failure';write(root/'summary.json',s)
    out=tmp_path/'report';report.generate(root,out)
    text=(out/'representation-gradient-diagnostics.md').read_text()
    assert '15 of 16 recorded objectives meet' in text
    assert 'Every objective nevertheless meets' not in text
    assert 'Unverified observations are retained' in text
    assert '## Measured observations' not in text  # A verified scientific interpretation must be gated.


def test_preflight_failure_does_not_invent_attempts_or_verification(tmp_path):
    root=tmp_path/'fixture';root.mkdir()
    config=json.loads((SOURCE/'config.json').read_text());write(root/'config.json',config)
    source=json.loads((SOURCE/'summary.json').read_text());source.update(status='failed',execution_valid=False,phase_A_status='unexecuted',next_step='resolve_verification_or_operational_failure')
    source['cases']={name:{'status':'blocked','executed':False,'verification_valid':False} for name in source['cases']}
    write(root/'summary.json',source);write(root/'phase-A.json',{'status':'unexecuted','subsets':{},'prediction_checks':{}})
    out=tmp_path/'report';report.generate(root,out);text=(out/'representation-gradient-diagnostics.md').read_text()
    assert '0 of 8 prespecified cases were attempted' in text
    assert '0 of 0 recorded objectives meet' in text
    assert 'Eight prespecified cases were attempted' not in text
    assert '## Measured observations' not in text
    assert 'All observed rigid ranks are6' not in text
