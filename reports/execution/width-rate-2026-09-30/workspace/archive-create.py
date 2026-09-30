from pathlib import Path
import json,hashlib,zipfile,shutil,subprocess
repo=Path('/Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex');raw=repo/'runs/aspirin-width-rate-2026-09-30';w=repo/'.superpowers/sdd/2026-09-30-width-rate';out=repo/'reports/execution/width-rate-2026-09-30'
if out.exists():raise ValueError('Do not overwrite an archive')
out.mkdir(parents=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((repo/'reports/width-rate-diagnostics.json').read_text());paths=sorted(p for p in raw.rglob('*') if p.is_file());rawhash={str(p.relative_to(raw)):digest(p) for p in paths}
assert rawhash==report['artifacts_sha256']
with zipfile.ZipFile(out/'raw-run.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in paths:z.write(p,str(p.relative_to(raw)))
with zipfile.ZipFile(out/'raw-run.zip') as z:
 assert z.testzip() is None and set(z.namelist())==set(rawhash)
 for name,h in rawhash.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
shutil.copytree(w,out/'workspace')
workspacehash={str(p.relative_to(w)):digest(p) for p in sorted(w.rglob('*')) if p.is_file()}
for name,h in workspacehash.items():assert digest(out/'workspace'/name)==h
repros={}
for directory in [Path('/private/tmp/width-rate-review-cwskh82s'),Path('/private/tmp/width-rate-review-start-ccda1a61')]:
 paths=sorted(p for p in directory.rglob('*') if p.is_file());hashes={str(p.relative_to(directory)):digest(p) for p in paths};dest=out/(directory.name+'.zip')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in paths:z.write(p,str(p.relative_to(directory)))
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None and set(z.namelist())==set(hashes)
  for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 repros[directory.name]={'original_path':str(directory),'file_count':len(paths),'files_sha256':hashes,'zip_sha256':digest(dest),'synthetic_engineering_metadata_not_research':True}
current={str(p.relative_to(repo)):digest(p) for p in [repo/'src/force_consistency/width_rate_runs.py',repo/'scripts/report_width_rate.py',repo/'src/force_consistency/width_rate_contract.py',repo/'src/force_consistency/cli.py']}
manifest={'archive':'width-rate-2026-09-30','production_revision':'18d250f06b9cf02ccfcf5637cf4c2f1b0c459e8e','raw_count':len(rawhash),'raw_files_sha256':rawhash,'raw_zip_sha256':digest(out/'raw-run.zip'),'raw_zip_bytes':(out/'raw-run.zip').stat().st_size,'workspace_count':len(workspacehash),'workspace_files_sha256':workspacehash,'review_reproductions':repros,'current_post_run_code_sha256':current,'preservation':'All original raw outputs, review folders and synthetic reproductions retained; supplemental archive only','research_rerun':False,'production_checkpoint_revalidated':False,'status':'failed_first_cell_three_blocked','verification':'ZIP CRC, exact entry set and SHA256 bytes; copied workspace bytes verified'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'raw_files':len(rawhash),'workspace_files':len(workspacehash),'raw_zip_bytes':manifest['raw_zip_bytes'],'raw_zip_sha256':manifest['raw_zip_sha256'],'manifest_sha256':digest(out/'manifest.json'),'synthetic_reproduction_files':sum(r['file_count'] for r in repros.values())},indent=2))
