from pathlib import Path
import subprocess,sys,json,hashlib
from datetime import datetime,timezone
r=Path('reports/execution/late-rate-2026-10-01')
command=[sys.executable,'-m','force_consistency.late_rate','--config','configs/aspirin-late-rate.json']
record={'command':command,'cwd':str(Path.cwd()),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':hashlib.sha256(Path(command[-1]).read_bytes()).hexdigest(),'numerical_source_sha256':hashlib.sha256(Path('src/force_consistency/late_rate.py').read_bytes()).hexdigest(),'maximum_invocations':1}
with (r/'production-invocation.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
with (r/'production-stdout.txt').open('x') as out,(r/'production-stderr.txt').open('x') as err:
 proc=subprocess.Popen(command,stdout=out,stderr=err)
 (r/'production-process.json').write_text(json.dumps({'pid':proc.pid,'status':'started'},indent=2)+'\n')
 code=proc.wait()
completion={'returncode':code,'ended_utc':datetime.now(timezone.utc).isoformat(),'invocation_sha256':hashlib.sha256((r/'production-invocation.json').read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256((r/'production-stdout.txt').read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256((r/'production-stderr.txt').read_bytes()).hexdigest()}
(r/'production-completion.json').write_text(json.dumps(completion,indent=2)+'\n');print(json.dumps(completion));sys.exit(code)
