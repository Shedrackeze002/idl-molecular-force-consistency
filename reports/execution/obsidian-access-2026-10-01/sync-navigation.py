"""Back up, append via authorized Obsidian CLI, and verify readback; no direct vault edits."""
from pathlib import Path
import hashlib,json,subprocess
from datetime import datetime,timezone
root=Path(__file__).parent
cli='/usr/local/bin/obsidian'
marker='## Current status — checkpoint-copy verification and CLI access correction'
repo=Path.cwd()
common='''
The separately reviewed checkpoint-copy verification passed once: the saved6000-update parameter/Adam state reproduced144prediction frames exactly, maximum difference0, with zero training updates/optimizer proposals/warmup. Final388tests passed253.19s; independent saved-result review found no material inconsistency. All145original raw files remain unchanged. Original width/rate matrix remains failed, both joint learning gates failed, three cells blocked, contrasts unavailable and full pilot pending. A successful copy does not retroactively validate the original trial or establish DFT accuracy/generalization/physical shape recovery.

Repository authority: reports/PAPER_EVIDENCE_LOG.md (WIDTHRATE-RECOVERY-FINDING-001, WIDTHRATE-RECOVERY-REVIEW-002, WIDTHRATE-RECOVERY-ARCHIVE-003), reports/width-rate-recovery-verification.json/md, reports/width-rate-recovery-execution-record.md, docs/reviews/2026-10-01-width-rate-recovery-final-review.md; verified preservation archive reports/execution/width-rate-recovery-2026-10-01. All failed/incomplete engineering checks and original/review folders retained. Next numerical milestone requires a separately prospective width/rate matrix plan and two independent reviews before execution.

CLI access correction: the executable and app were available. The restricted Codex sandbox could not detect Obsidian and returned a misleading application-not-found message; permitted host execution successfully read these notes. Earlier CLI-unavailable/pending-navigation statements are superseded by this verified synchronization. No installation, app settings or CLI registration changed. Use the authorized host-execution path for future Obsidian CLI operations; repository remains numerical authority.
'''
notes={'hub':'Codex/IDL MLIP Project/Project Hub.md','findings':'Codex/IDL MLIP Project/Research Findings.md'}
records={}
def call(*args):
 r=subprocess.run([cli,'vault=hermes',*args],capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError(r.stderr or r.stdout)
 return r.stdout
for label,path in notes.items():
 before=call('read','path='+path);(root/(label+'-before.md')).write_text(before)
 content=marker+'\n\nUpdated '+datetime.now(timezone.utc).isoformat()+'. This current-state section supersedes older running/planned/pending-navigation summaries.\n'+common
 if marker not in before:call('append','path='+path,'content='+content)
 after=call('read','path='+path);(root/(label+'-after.md')).write_text(after)
 assert before.strip() in after and marker in after and common.strip() in after
 records[label]={'path':path,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'append_verified':True}
record={'at_utc':datetime.now(timezone.utc).isoformat(),'diagnosis':'same CLI help fails in sandbox (exit1); host help and project read succeed (exit0)','vault':'hermes','method':'CLI backup, append, readback; no direct vault write','notes':records}
(root/'verification.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(record,indent=2))
