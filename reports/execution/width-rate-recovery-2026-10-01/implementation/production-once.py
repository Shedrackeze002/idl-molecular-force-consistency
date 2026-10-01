"""One explicit invocation; refuse reused marker/root. Controller owns its 180s cap."""
import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from force_consistency.width_rate_contract import digest,read
from force_consistency.width_rate_recovery_contract import verify_recovery_sources
work=Path(__file__).parent;config=read('configs/aspirin-width-rate-recovery.json');verify_recovery_sources(config)
assert not Path(config['run_directory']).exists()
log=work/'full-suite.txt';text=log.read_text();assert 'failed' not in text and 'passed in' in text and '[100%]' in text
record={'started_at_utc':datetime.now(timezone.utc).isoformat(),'full_suite_log_sha256':digest(log),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'command':[sys.executable,'-m','force_consistency.width_rate_recovery','--config','configs/aspirin-width-rate-recovery.json'],'backend':'cpu','maximum_invocations':1}
with (work/'production-invocation.json').open('x') as f:json.dump(record,f,indent=2)
completed=subprocess.run(record['command'],env={**os.environ,'JAX_PLATFORMS':'cpu'},capture_output=True)
(work/'production-stdout.txt').write_bytes(completed.stdout);(work/'production-stderr.txt').write_bytes(completed.stderr)
record.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),exit_code=completed.returncode,stdout_sha256=digest(work/'production-stdout.txt'),stderr_sha256=digest(work/'production-stderr.txt'))
(work/'production-invocation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(completed.stdout.decode(errors='replace'));print('Invocation exit:',completed.returncode)
raise SystemExit(completed.returncode)
