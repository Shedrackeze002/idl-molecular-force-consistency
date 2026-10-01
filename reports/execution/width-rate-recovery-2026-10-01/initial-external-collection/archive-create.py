"""Supplemental archive; preserve originals and verify every archived byte."""
from pathlib import Path
import hashlib,json,shutil,subprocess,zipfile

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes(root):return {str(p.relative_to(root)):digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and not p.is_symlink()}
def archive_zip(source,target):
    expected=hashes(source)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in expected:z.write(Path(source)/name,name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(expected)
        assert {name:hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}==expected
    return {'source':str(source),'file_count':len(expected),'files_sha256':expected,'zip_sha256':digest(target),'bytes':target.stat().st_size}

work=Path(__file__).parent;target=Path('reports/execution/width-rate-recovery-2026-10-01');target.mkdir(parents=True,exist_ok=True)
run=Path('runs/aspirin-width-rate-recovery-2026-10-01');result=json.loads(Path('reports/width-rate-recovery-verification.json').read_text());production=json.loads((work/'production-invocation.json').read_text());revision=production['revision']
code=target/'production-code';names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'src','scripts/report_width_rate_recovery.py','pyproject.toml','uv.lock'],text=True).splitlines()
for name in names:
    path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(subprocess.check_output(['git','show',revision+':'+name]))
for name,want in json.loads((run/'prospective.json').read_text())['code']['files_sha256'].items():assert digest(code/name)==want
manifest={'role':result['role'],'recovery_status':result['status'],'verification_valid':result['verification_valid'],'original_status':'failed','full_pilot':'pending','production_revision':revision,'code_files_sha256':hashes(code),'raw':archive_zip(run,target/'raw.zip'),'workspaces':{},'synthetic_reproductions':{},'retention':'All original raw/workspaces/reviewer repros preserved; archive supplements originals. Synthetic engineering records are not research trials.'}
for label,source in [('implementation',work),('planning',Path('.superpowers/sdd/2026-10-01-width-rate-recovery-plan'))]:
    dest=target/label
    for name,want in hashes(source).items():
        out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,out);assert digest(out)==want
    assert hashes(dest)==hashes(source);manifest['workspaces'][label]={'source':str(source),'file_count':len(hashes(source)),'files_sha256':hashes(source)}
source=Path('/private/tmp/width-rate-recovery-review-b2ef4b0')
if source.exists():manifest['synthetic_reproductions']['reviewer']=archive_zip(source,target/'synthetic-reviewer.zip')
external=target/'synthetic-unowned-root-failures';external.mkdir(exist_ok=True)
for p in sorted(Path('/private/tmp').glob('width-rate-recovery-unowned-*.json')):
    shutil.copyfile(p,external/p.name);assert digest(p)==digest(external/p.name)
manifest['synthetic_reproductions']['external_failures']={'file_count':len(hashes(external)),'files_sha256':hashes(external)}
manifest['reports_sha256']={name:digest(name) for name in ('reports/width-rate-recovery-verification.json','reports/width-rate-recovery-verification.md','reports/width-rate-recovery-execution-record.md','docs/reviews/2026-10-01-width-rate-recovery-final-review.md')}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'manifest_sha256':digest(target/'manifest.json'),'raw_file_count':manifest['raw']['file_count'],'raw_zip_sha256':manifest['raw']['zip_sha256'],'workspace_counts':{k:v['file_count'] for k,v in manifest['workspaces'].items()},'reviewer_reproduction_count':manifest['synthetic_reproductions'].get('reviewer',{}).get('file_count'),'synthetic_external_failure_count':manifest['synthetic_reproductions']['external_failures']['file_count']},indent=2))
