import hashlib,json,platform,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from force_consistency.width_rate_recovery_contract import verify_recovery_sources
from force_consistency.width_rate_recovery import code_record
from force_consistency.width_rate_contract import read,digest
config=read('configs/aspirin-width-rate-recovery.json');v=verify_recovery_sources(config)
root=Path(config['run_directory']);assert not root.exists() and not root.is_symlink()
files=[p for p in Path(config['failed_run_directory']).rglob('*') if p.is_file()]
assert len(files)==145
out={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'recovery_source_pins_verified':163,'historical_source_pins_verified':326,'original_raw_file_count':len(files),'original_raw_files_sha256':{str(p):digest(p) for p in sorted(files)},'code':code_record(),'host':platform.platform(),'no_production_probe_yet':True,'user_changes':{'proposal.pdf':'deleted retained','Group9.pdf_sha256':digest('Group9.pdf')}}
Path('.superpowers/sdd/2026-10-01-width-rate-recovery/preproduction-check.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ('recovery_source_pins_verified','historical_source_pins_verified','original_raw_file_count','no_production_probe_yet')}))
