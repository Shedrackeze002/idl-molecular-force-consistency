from pathlib import Path
import subprocess,json,hashlib
r=Path('reports/execution/late-rate-2026-10-01');report=json.loads(Path('reports/late-rate-diagnostics.json').read_text())
text='\n\n## Current status — late learning-rate diagnostic outcome\n\nThis section supersedes the preproduction note. Actual once-only experiment finished; saved execution validity: '+str(report['execution_valid'])+'. Full pilot remains pending.\n\n'
for name,a in report['arms'].items():
 if a['execution_valid']:
  f=a['original_baseline_fractions']['fitting'];text+=name+': fitting energy/force fractions '+str(f['energy_mse_eV2'])+' / '+str(f['force_mse_eV2_per_A2'])+'; original fitting-and-fitted-anchor joint gate '+str(a['learning_accepted'])+'.\n\n'
 else:text+=name+': incomplete/invalid; preserved raw summary and unknown/partial work, no complete contrast.\n\n'
text+='New computed updates '+str(report['additional_updates'])+'; fitting exposures '+str(report['additional_frame_exposures'])+'. Both schedules inherit the same update-6,000 state and Adam moments; fixed 4,000 extra updates each, 128 fitted frames and 16 fully fitted anchors, seed 0, CPU float64. No test evaluation, GPU run or main consistency/shape-recovery result. One bundled late-decay schedule and single chosen starting trajectory do not prove optimality, convergence or generalization.\n\nRepository authority: reports/late-rate-diagnostics.json/md, PAPER_EVIDENCE_LOG.md LATE-RATE-* entries and reports/execution/late-rate-2026-10-01. All negative/engineering/review outcomes retained; Obsidian is navigation only.\n'
records={}
for label,name in [('hub','Project Hub.md'),('findings','Research Findings.md')]:
 path='Codex/IDL MLIP Project/'+name
 before=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 if 'unable to find Obsidian' in before.stdout:raise RuntimeError('CLI unavailable')
 (r/f'obsidian-{label}-before-final.md').write_text(before.stdout)
 subprocess.run(['/usr/local/bin/obsidian','vault=hermes','append','path='+path,'content='+text],check=True,capture_output=True,text=True)
 after=subprocess.run(['/usr/local/bin/obsidian','vault=hermes','read','path='+path],check=True,capture_output=True,text=True)
 (r/f'obsidian-{label}-final.md').write_text(after.stdout)
 if text.strip() not in after.stdout:raise RuntimeError('Readback mismatch')
 records[label]={'status':'verified','backup_sha256':hashlib.sha256(before.stdout.encode()).hexdigest(),'readback_sha256':hashlib.sha256(after.stdout.encode()).hexdigest()}
(r/'obsidian-final-verification.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
