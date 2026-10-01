from pathlib import Path
import subprocess,json,hashlib
r=Path('reports/execution/late-rate-2026-10-01')
content='\n\n## Current status — late learning-rate lever, preproduction\n\nThe next controlled experiment is authorized. Two independent amended plan reviews passed. Fresh implementation review found four Important reporting/accounting/lineage gaps, preserved and repaired;45focused checks passed. Final full regression/follow-up gates remain pending at this update. No new production training yet.\n\nFrozen experiment: two4000-update continuations from identical verified width32/.001 update6000 parameters and Adam moments, constant.001 versus late.0003 through8000 then.0001. Same128 fitted/16 fully fitted anchors, seed0CPUfloat64, balanced loss and original joint acceptance criteria. No test evaluation, GPU trial or full pilot.\n\nSaved-only residual analysis found endpoint energy MSE dominated by centered residual variance; common squared mean error is tiny. This does not explain the historical spike or amend the schedule. Repository authority: PAPER_EVIDENCE_LOG.md LATE-RATE-PLAN/REVIEW/SAVED-OBSERVATIONS/ENGINEERING entries, docs/superpowers/plans/2026-10-01-late-rate.md, reports/execution/late-rate-2026-10-01. Obsidian is navigation only.\n'
records={}
for label,name in [('hub','Project Hub.md'),('findings','Research Findings.md')]:
 path='Codex/IDL MLIP Project/'+name
 before=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 if 'unable to find Obsidian' in before.stdout:raise RuntimeError('CLI app unavailable')
 (r/f'obsidian-{label}-before.md').write_text(before.stdout)
 subprocess.run(['/usr/local/bin/obsidian','vault=hermes','append','path='+path,'content='+content],check=True,capture_output=True,text=True)
 after=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 (r/f'obsidian-{label}-preproduction.md').write_text(after.stdout)
 if content.strip() not in after.stdout:raise RuntimeError('CLI readback did not match')
 records[label]={'backup_sha256':hashlib.sha256(before.stdout.encode()).hexdigest(),'readback_sha256':hashlib.sha256(after.stdout.encode()).hexdigest(),'status':'verified'}
(r/'obsidian-preproduction-verification.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
