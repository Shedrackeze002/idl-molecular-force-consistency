import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

repo=Path.cwd()
target=repo/'reports/evidence/2026-11-21/deadline-integration-review'
target.mkdir(parents=True,exist_ok=True)
previous='3def93488965f7d7a55064175067bf5f313cdf3d'
candidate='f550073eae1bd1b880ff04f8daee587c5412840b'
def blob(revision,name):
    return subprocess.check_output(['git','--no-replace-objects','cat-file','blob',revision+':'+name])
test_tree=ast.parse(blob(candidate,'tests/test_mace_trusted_launcher.py'))
nodes=[node for node in test_tree.body if isinstance(node,(ast.Import,ast.ImportFrom)) or
       isinstance(node,ast.FunctionDef) and node.name in {'sha','git','boundary','run'}]
records=[]
with tempfile.TemporaryDirectory(prefix='mace-setup-timing-') as scratch:
    for label,revision in [('previous',previous),('candidate',candidate)]:
        source=Path(scratch)/label
        launcher=source/'trust/mace_trusted_launcher.py'
        launcher.parent.mkdir(parents=True)
        launcher.write_bytes(blob(revision,'trust/mace_trusted_launcher.py'))
        spec=importlib.util.spec_from_file_location('timing_inventory_'+label,launcher)
        trusted=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trusted)
        for name in (*trusted.IMPLEMENTATION_FILES,*trusted.FROZEN_FILES):
            path=source/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(blob(revision,name))
        namespace={'SOURCE':launcher,'trusted':trusted}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_test_fixture','exec'),namespace)
        base=Path(scratch)/(label+'-case')
        base.mkdir()
        request=types.SimpleNamespace(param={'supervisor_edits':[
            ('WALL_SECONDS = 1800','WALL_SECONDS = .2')],
            'launcher_edits':[('sources = verify_sources(checkout, commit)',
                'time.sleep(1.25)\n        sources = verify_sources(checkout, commit)')]})
        case=namespace['boundary'].__wrapped__(base,request)
        started=time.monotonic()
        result=namespace['run'](case)
        elapsed=time.monotonic()-started
        status=json.loads((case['attempt']/'supervisor-status.json').read_text())
        records.append({'revision':revision,'kind':label,'budget_seconds':.2,'injected_delay_seconds':1.25,
            'elapsed_wall_seconds':elapsed,'exit_code':result.returncode,'status':status['status'],
            'failure_stage':status.get('failure_stage'),'worker_marker_exists':case['marker'].exists(),
            'reported_elapsed_seconds':status.get('elapsed_seconds')})
comparison={'status':'EXPECTED_OVERRUN_REPRODUCED_AND_FIXED','runs':records,
    'execution_boundary':'Synthetic source fixtures only; no calculator or checkpoint/model execution.'}
valid=(records[0]['elapsed_wall_seconds']>=1.25 and records[1]['elapsed_wall_seconds']<.85
       and all(r['exit_code']==2 and r['status']=='TIMEOUT' and not r['worker_marker_exists'] for r in records))
if not valid: comparison['status']='UNEXPECTED_RESULT'
(target/'setup-deadline-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
raise SystemExit(0 if valid else 1)
