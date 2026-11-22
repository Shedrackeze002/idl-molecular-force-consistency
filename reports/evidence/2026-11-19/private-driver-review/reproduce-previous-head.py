import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

repo = Path.cwd()
target = repo / 'reports/evidence/2026-11-19/private-driver-review'
target.mkdir(parents=True, exist_ok=True)
prior = '83842ef97b112b5d88d5cde5ef4f033932462923'
spec = importlib.util.spec_from_file_location('source_inventory', repo / 'trust/mace_trusted_launcher.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)

with tempfile.TemporaryDirectory(prefix='mace-previous-review-') as scratch:
    previous = Path(scratch)
    for name in (*inventory.IMPLEMENTATION_FILES, *inventory.FROZEN_FILES):
        raw = subprocess.check_output(['git', '--no-replace-objects', 'cat-file', 'blob', prior + ':' + name])
        path = previous / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    tests = previous / 'tests/test_mace_trusted_launcher.py'
    content = tests.read_text().replace(
        '[signal.SIGTERM, signal.SIGINT, signal.SIGHUP]',
        '[signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT]')
    content += '''

def test_reported_forgery_regression(tmp_path):
    marker = tmp_path / 'forged-parity'
    bootstrap = SOURCE.parent.parent / 'scripts/mace_bootstrap.py'
    probe = ("import importlib.util,sys; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('probe',sys.argv.pop(1)); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "module.configure_imports(); import mace_loader as loader; "
        "loader._worker_capability=lambda *args: None; import check_mace_parity as worker; "
        f"worker.run_parity=lambda *args: (Path({str(marker)!r}).touch(),{{'status':'PARITY_PASS'}})[1]; "
        "worker.main()")
    result = subprocess.run([sys.executable,'-I','-S','-c',probe,str(bootstrap),
        '--packet-root',str(tmp_path),'--attempt-root',str(tmp_path/'attempt')],
        text=True,capture_output=True,timeout=30)
    assert result.returncode != 0, 'forged public main returned success'
    assert not marker.exists(), 'forged public main executed parity marker'


def test_reported_initial_receipt_regression(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('old_supervisor', SOURCE.parent.parent/'scripts/supervise_mace_parity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.Path.write_text
    fired = False
    def interrupt(path, data, *args, **kwargs):
        nonlocal fired
        result = original(path,data,*args,**kwargs)
        if path.name=='supervisor-status.json' and not fired:
            fired=True
            os.kill(os.getpid(),signal.SIGTERM)
        return result
    monkeypatch.setattr(module.Path,'write_text',interrupt)
    output=tmp_path/'attempt'
    try:
        module.supervise(tmp_path,output,command=[sys.executable,'-c','pass'])
    except module.SupervisorInterrupted:
        pass
    status=json.loads((output/'supervisor-status.json').read_text())
    assert status['status']=='INTERRUPTED', 'initial receipt remained RUNNING'
'''
    tests.write_text(content)
    selection = 'test_reported_forgery or test_reported_initial or test_parent_interruption_reaps_actual_handoff_worker'
    result = subprocess.run([sys.executable,'-m','pytest','-q',str(tests),'-k',selection,'-p','no:cacheprovider'],
        capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(previous/'scripts')),cwd=previous)
    def redact(value):
        cleaned = re.sub(r'/tmp/pytest-of-[^/\s]+', '$PYTEST_TMP',
            value.replace(scratch,'$PREVIOUS_SOURCE').replace(str(repo),'$REPO_ROOT')
            .replace(str(Path(sys.executable).parent.parent.parent),'$ONBOARDING_ROOT'))
        return re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
    for kind, value in [('stdout',result.stdout),('stderr',result.stderr),('exit',str(result.returncode)+'\n')]:
        (target/('previous-head-reproductions.'+kind+'.txt')).write_text(redact(value))
    expected = result.returncode == 1 and '3 failed, 3 passed' in result.stdout
    summary = {'status':'EXPECTED_FAILURES_REPRODUCED' if expected else 'UNEXPECTED_RESULT',
        'previous_commit':prior, 'pytest_exit_code':result.returncode,
        'expected_failures':['forged module callback','SIGQUIT detached worker','interrupted initial RUNNING receipt'],
        'passing_controls':['SIGTERM','SIGINT','SIGHUP'],
        'execution_boundary':'Synthetic markers and disposable children only; no checkpoint/model execution.'}
    (target/'previous-head-reproductions.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not expected:
        print(redact(result.stdout[-12000:]))
    raise SystemExit(0 if expected else 1)
