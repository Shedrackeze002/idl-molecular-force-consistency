from pathlib import Path
import subprocess,json,hashlib
p=Path('reports/execution/capacity-2026-10-02');text=(p/'obsidian-planned-note.md').read_text();records={}
for label,name in [('hub','Project Hub.md'),('findings','Research Findings.md')]:
 path='Codex/IDL MLIP Project/'+name
 before=subprocess.run(['/usr/local/bin/obsidian','read','vault=hermes','path='+path],check=True,capture_output=True,text=True).stdout
 if before.startswith('Error:') or not before.strip():raise RuntimeError('CLI read failed')
 (p/f'obsidian-{label}-before-plan.md').write_text(before)
 out=subprocess.run(['/usr/local/bin/obsidian','append','vault=hermes','path='+path,'content='+text],check=True,capture_output=True,text=True).stdout
 if out.startswith('Error:'):raise RuntimeError('CLI append failed: '+out)
 after=subprocess.run(['/usr/local/bin/obsidian','read','vault=hermes','path='+path],check=True,capture_output=True,text=True).stdout
 (p/f'obsidian-{label}-planned.md').write_text(after)
 if text.strip() not in after:raise RuntimeError('CLI readback mismatch')
 records[label]={'status':'verified','before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest()}
(p/'obsidian-plan-verification.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
