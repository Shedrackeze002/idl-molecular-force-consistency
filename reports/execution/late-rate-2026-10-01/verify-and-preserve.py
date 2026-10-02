"""Saved-only verification and lossless archive; never launches training."""
from pathlib import Path
import json,hashlib,zipfile
from force_consistency.late_rate import build_report,read,digest,atomic_json
r=Path('reports/execution/late-rate-2026-10-01');root=Path('runs/aspirin-late-rate-2026-10-01')
actual=build_report(root);saved=read('reports/late-rate-diagnostics.json')
if actual!=saved:raise RuntimeError('Fresh saved-only report differs from final saved report')
if not actual['execution_valid']:raise RuntimeError('Execution/report validation failed; preserve failures separately')
checks={'status':'passed','report_exactly_reproduced':True,'production_invocations':1,'raw_artifact_count':len(read(root/'raw-artifacts.json')),'source_pins_unchanged':True,'additional_updates':actual['additional_updates'],'additional_frame_exposures':actual['additional_frame_exposures'],'closure_sha256':digest(root/'finalization.json'),'report_sha256':digest('reports/late-rate-diagnostics.json')}
archive=r/'numerical-run.zip'
if archive.exists():raise FileExistsError('Archive already exists')
files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in files:z.write(root/name,name)
with zipfile.ZipFile(archive) as z:
 if z.testzip() is not None or set(z.namelist())!=set(files):raise RuntimeError('Archive CRC/inventory mismatch')
 for name,sha in files.items():
  if hashlib.sha256(z.read(name)).hexdigest()!=sha:raise RuntimeError('Archived bytes differ: '+name)
checks.update(archive_sha256=digest(archive),archive_file_count=len(files),archive_crc_verified=True,archive_bytes_verified=True,original_folders_retained=True)
atomic_json(r/'preservation-verification.json',checks);atomic_json(r/'numerical-run-inventory.json',files)
print(json.dumps(checks))
