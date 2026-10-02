from pathlib import Path
import json,subprocess,hashlib
p=Path('reports/execution/late-rate-2026-10-01');text=(p/'obsidian-paper-observation.md').read_text();result={}
for label,name in [('hub','Project Hub.md'),('findings','Research Findings.md')]:
 path='Codex/IDL MLIP Project/'+name
 before=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 if 'unable to find Obsidian' in before.stdout:raise RuntimeError('CLI unavailable')
 (p/f'obsidian-{label}-before-observation.md').write_text(before.stdout)
 subprocess.run(['/usr/local/bin/obsidian','vault=hermes','append','path='+path,'content='+text],check=True,capture_output=True,text=True)
 after=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 (p/f'obsidian-{label}-observation.md').write_text(after.stdout)
 if text.strip() not in after.stdout:raise RuntimeError('CLI readback mismatch')
 result[label]={'status':'verified','before_sha256':hashlib.sha256(before.stdout.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.stdout.encode()).hexdigest()}
(p/'obsidian-observation-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
