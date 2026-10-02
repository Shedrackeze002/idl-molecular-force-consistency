"""CLI-only notes update with retained backups, exact text detection and readback."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=Path('reports/execution/capacity-2026-10-02');phase=sys.argv[1];text=Path(sys.argv[2]).read_text();records={}
if not phase.replace('-','').isalnum():raise ValueError('Unsafe phase')
def call(command,**kwargs):
    args=['/usr/local/bin/obsidian',command,'vault=hermes']+[k+'='+v for k,v in kwargs.items()]
    out=subprocess.run(args,check=True,capture_output=True,text=True).stdout
    if out.startswith('Error:'):raise RuntimeError(out)
    return out
try:
    for label,name in [('hub','Project Hub.md'),('findings','Research Findings.md')]:
        path='Codex/IDL MLIP Project/'+name;before=call('read',path=path)
        if not before.strip():raise RuntimeError('Empty CLI note')
        backup=p/f'obsidian-{label}-before-{phase}.md'
        with backup.open('x') as f:f.write(before)
        if text.strip() not in before:call('append',path=path,content=text)
        after=call('read',path=path)
        with (p/f'obsidian-{label}-{phase}.md').open('x') as f:f.write(after)
        if text.strip() not in after:raise RuntimeError('CLI readback mismatch')
        records[label]={'status':'verified','before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256':hashlib.sha256(after.encode()).hexdigest()}
except Exception as exc:
    records['failure']={'error':str(exc)};raise
finally:
    (p/f'obsidian-{phase}-verification.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records))
