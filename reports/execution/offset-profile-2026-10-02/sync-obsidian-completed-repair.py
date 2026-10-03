"""Retained completion-sync recovery: explicit vault prefix and fail-closed CLI reads."""
import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).parent
phase='completed-recovered'
text=(p/'obsidian-completed-text.md').read_text()
records={}
def call(command,**kwargs):
    args=['/usr/local/bin/obsidian','vault=hermes',command]+[k+'='+v for k,v in kwargs.items()]
    proc=subprocess.run(args,check=True,capture_output=True,text=True,timeout=20)
    out=proc.stdout
    if not out.strip() or out.startswith('Error:') or out.strip() in ('Vault not found.','File not found.'):
        raise RuntimeError(command+' failed: '+out.strip())
    return out
try:
    notes=[('hub','Project Hub.md'),('findings','Research Findings.md')]
    before={}
    for label,name in notes:
        path='Codex/IDL MLIP Project/'+name
        value=call('read',path=path)
        if not value.lstrip().startswith('#') or len(value)<100:raise RuntimeError('Not verified project-note content: '+label)
        before[label]=value
        with (p/f'obsidian-{label}-before-{phase}.md').open('x') as f:f.write(value)
    for label,name in notes:
        path='Codex/IDL MLIP Project/'+name
        if text.strip() not in before[label]:
            content=text.replace('\\','\\\\').replace('\n','\\n')
            call('append',path=path,content=content)
        after=call('read',path=path)
        with (p/f'obsidian-{label}-{phase}.md').open('x') as f:f.write(after)
        if text.strip() not in after:raise RuntimeError('Exact CLI readback mismatch: '+label)
        records[label]={'status':'verified','before_sha256':hashlib.sha256(before[label].encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'exact_sync_text_present':True}
except Exception as exc:
    records['failure']={'error':str(exc)}
    raise
finally:
    with (p/f'obsidian-{phase}-verification.json').open('x') as f:json.dump(records,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(records))
