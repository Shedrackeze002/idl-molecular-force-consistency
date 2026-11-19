import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

repo = Path.cwd()
target = repo/'reports/evidence/2026-11-19/closed-runtime-review'
target.mkdir(parents=True,exist_ok=True)
prior = 'f812be6a40c9008715685cee8dfdea24245a8f4f'
def blob(name):
    return subprocess.check_output(['git','--no-replace-objects','cat-file','blob',prior+':'+name])
with tempfile.TemporaryDirectory(prefix='mace-old-runtime-') as scratch:
    old = Path(scratch)
    launcher = old/'trust/mace_trusted_launcher.py'
    launcher.parent.mkdir()
    launcher.write_bytes(blob('trust/mace_trusted_launcher.py'))
    spec=importlib.util.spec_from_file_location('old_inventory',launcher)
    inventory=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    for name in (*inventory.IMPLEMENTATION_FILES,*inventory.FROZEN_FILES):
        path=old/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(blob(name))
    candidate='73787b0cd61a92b0041a408e4dfaf12e412d4806'
    test_source=subprocess.check_output(['git','--no-replace-objects','cat-file','blob',
        candidate+':tests/test_mace_trusted_launcher.py'])
    tree=ast.parse(test_source)
    names=['test_forged_supervisor_callback_with_unapproved_receipts_never_runs',
           'test_direct_internal_loader_refuses_injected_constructor_before_any_model_access']
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    test=old/'tests/test_reported_boundaries.py'
    preamble='import importlib.util,json,os,subprocess,sys\nfrom pathlib import Path\nimport pytest\n'
    preamble+=f'SOURCE=Path({str(launcher)!r})\n'
    test.write_text(preamble+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n')
    selection=[str(test)+'::'+names[0]+'[main]',str(test)+'::'+names[1]]
    result=subprocess.run([sys.executable,'-m','pytest','-q',*selection,'-p','no:cacheprovider'],
        capture_output=True,text=True,cwd=old,env=dict(os.environ,PYTHONPATH=str(old/'scripts')))
    def redact(value):
        value=value.replace(scratch,'$PREVIOUS_SOURCE').replace(str(repo),'$REPO_ROOT')
        value=value.replace(str(Path(sys.executable).parent.parent.parent),'$ONBOARDING_ROOT')
        value=re.sub(r'/tmp/pytest-of-[^/\s]+','$PYTEST_TMP',value)
        return re.sub(r'[ \t]+$','',value,flags=re.MULTILINE)
    for kind,value in [('stdout',result.stdout),('stderr',result.stderr),('exit',str(result.returncode)+'\n')]:
        (target/('previous-head-reproductions.'+kind+'.txt')).write_text(redact(value))
    expected=result.returncode==1 and '2 failed' in result.stdout
    summary={'status':'EXPECTED_FAILURES_REPRODUCED' if expected else 'UNEXPECTED_RESULT',
        'previous_commit':prior,'pytest_exit_code':result.returncode,
        'expected_failures':['forged supervisor callback with NOT_APPROVED/hash-mismatched receipts',
                             'direct loader invokes an injected constructor outside supervisor'],
        'execution_boundary':'Synthetic process-like object and non-model bytes only; no real model construction or inference.'}
    (target/'previous-head-reproductions.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not expected: print(redact(result.stdout[-10000:]))
    raise SystemExit(0 if expected else 1)
