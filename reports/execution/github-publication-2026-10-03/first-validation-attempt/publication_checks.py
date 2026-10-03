"""Retained publication operations/checks. Never mutate the research Git metadata."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex')
ROOT = Path('/private/tmp/idl-lfs-publishing-2026-10-03-17aa83c')
PUB = ROOT / 'publisher'
VERIFY = ROOT / 'remote-verification'
RECORD = SOURCE / 'reports/execution/github-publication-2026-10-03'
OLD = '17aa83c751f95f5a1aef54fd1974340bfc5a1527'
BASE = '663bc49640901904530c9a2ede3903cf43c87b9e'
URL = 'https://github.com/Shedrackeze002/idl-molecular-force-consistency.git'
BRANCH = 'codex/project-handoff'
LFSBIN = ROOT / 'tools/extracted/git-lfs-3.8.0'
ENV = dict(os.environ, PATH=str(LFSBIN) + os.pathsep + os.environ['PATH'], GIT_OPTIONAL_LOCKS='0')
MANIFEST = json.loads((RECORD / 'large-artifacts.json').read_text())
ARTS = {a['path']: a for a in MANIFEST['artifacts']}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def filehash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    (RECORD / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, env=ENV)

def command(label, repo, args, env=None):
    result = subprocess.run(args, cwd=repo, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    save(label + '.json', {'argv': args, 'cwd': str(repo), 'exit_code': result.returncode, 'output': result.stdout.decode(errors='replace')})
    print(label, 'exit', result.returncode, flush=True)
    if result.returncode:
        print(result.stdout.decode(errors='replace')[-4000:], flush=True)
        raise RuntimeError(label)
    return result.stdout

def tree(repo, revision):
    entries = {}
    for raw in git(repo, 'ls-tree', '-r', '-z', revision).split(b'\0'):
        if raw:
            meta, path = raw.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            entries[path.decode()] = [mode, kind, oid]
    return entries

def snapshot():
    tracked = tree(SOURCE, OLD)
    allowed = lambda p: p.startswith('reports/execution/github-publication-2026-10-03/') or p == 'docs/team/LFS_PUBLICATION_PLAN.md' or p.startswith('docs/team/reviews/2026-10-03-lfs-') or p == 'docs/team/LFS_SETUP.md'
    working = {p: (filehash(SOURCE / p) if (SOURCE / p).is_file() else None) for p in tracked if p != 'reports/PAPER_EVIDENCE_LOG.md'}
    untracked = git(SOURCE, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    working.update({p: filehash(SOURCE / p) for p in untracked if p and not allowed(p) and (SOURCE / p).is_file()})
    status = '\n'.join(line for line in git(SOURCE, 'status', '--porcelain', '--untracked-files=all').decode().splitlines() if not allowed(line[3:]) and line[3:] != 'reports/PAPER_EVIDENCE_LOG.md')
    log = SOURCE / 'reports/PAPER_EVIDENCE_LOG.md'
    return {'head': git(SOURCE, 'rev-parse', 'HEAD').decode().strip(), 'branch': git(SOURCE, 'branch', '--show-current').decode().strip(), 'refs': git(SOURCE, 'show-ref').decode(), 'local_config_sha256': filehash(SOURCE / '.git/config'), 'index_sha256': filehash(SOURCE / '.git/index'), 'status_outside_allowlist': status, 'working_sha256': working, 'log_size': log.stat().st_size, 'log_sha256': filehash(log)}

def baseline():
    s = snapshot()
    assert s['head'] == OLD and s['branch'] == 'codex/landscape17-baseline'
    for p, a in ARTS.items():
        assert (SOURCE / p).stat().st_size == a['size_bytes'] and filehash(SOURCE / p) == a['sha256']
    save('source-preservation-baseline.json', s)
    print('Source baseline and four original payload hashes verified', flush=True)

def isolation(repo):
    assert not (repo / '.git/objects/info/alternates').exists()
    originals = {(p.stat().st_dev, p.stat().st_ino) for p in (SOURCE / '.git/objects').rglob('*') if p.is_file()}
    assert all((p.stat().st_dev, p.stat().st_ino) not in originals for p in (repo / '.git/objects').rglob('*') if p.is_file())

def setup(repo, storage):
    git(repo, 'config', '--local', 'core.hooksPath', str(repo / '.git/hooks'))
    git(repo, 'config', '--local', 'lfs.storage', str(storage))
    git(repo, 'config', '--local', 'lfs.url', URL + '/info/lfs')
    git(repo, 'config', '--local', 'lfs.pushurl', URL + '/info/lfs')
    command(repo.name + '-lfs-install', repo, ['git', 'lfs', 'install', '--local'])
    hook = repo / '.git/hooks/pre-push'
    assert os.access(hook, os.X_OK) and 'git lfs pre-push' in hook.read_text()
    command(repo.name + '-lfs-environment', repo, ['git', 'lfs', 'env'])

def clone():
    assert not PUB.exists()
    command('independent-source-clone', ROOT, ['git', 'clone', '--no-local', '--no-hardlinks', '--no-checkout', '--single-branch', '--branch', 'codex/landscape17-baseline', str(SOURCE), str(PUB)])
    isolation(PUB)
    git(PUB, 'checkout', '-b', BRANCH, OLD)
    git(PUB, 'remote', 'set-url', 'origin', URL)
    command('fetch-main', PUB, ['git', 'fetch', 'origin', 'refs/heads/main:refs/remotes/origin/main'])
    assert git(PUB, 'rev-parse', 'refs/remotes/origin/main').decode().strip() == BASE
    git(PUB, 'merge-base', '--is-ancestor', BASE, 'HEAD')
    for key in ['user.name', 'user.email']:
        value = git(SOURCE, 'config', '--get', key).decode().strip()
        assert value
        git(PUB, 'config', '--local', key, value)
    command('publisher-inherited-config-names', PUB, ['git', 'config', '--show-origin', '--name-only', '--get-regexp', '^(core\\.hooksPath|filter\\.|lfs\\.)']) if subprocess.run(['git','config','--get-regexp','^(core\\.hooksPath|filter\\.|lfs\\.)'],cwd=PUB,env=ENV,stdout=subprocess.DEVNULL).returncode == 0 else save('publisher-inherited-config-names.json', {'status':'no inherited hook/filter/lfs settings'})
    setup(PUB, PUB / '.git/lfs')
    assert not git(PUB, 'status', '--porcelain').strip()
    save('publisher-isolation.json', {'independent_objects': True, 'no_alternates': True, 'source_head': OLD, 'base': BASE, 'origin': URL, 'branch': BRANCH, 'lfs_version': git(PUB, 'lfs', 'version').decode().strip()})

def migrate():
    assert git(PUB, 'rev-parse', 'HEAD').decode().strip() == OLD and not git(PUB, 'status', '--porcelain').strip()
    command('lfs-migration', PUB, ['git', 'lfs', 'migrate', 'import', '--include-ref=refs/heads/' + BRANCH, '--exclude-ref=refs/remotes/origin/main', '--include=' + ','.join(ARTS), '--object-map=' + str(RECORD / 'raw-rewritten-commits.csv')])

def pointer(repo, oid, artifact):
    raw = git(repo, 'cat-file', 'blob', oid)
    expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{artifact['sha256']}\nsize {artifact['size_bytes']}\n".encode()
    assert raw == expected, (oid, raw[:200])

def validate():
    oldcommits = git(SOURCE, 'rev-list', '--reverse', OLD, '^' + BASE).decode().splitlines()
    assert len(oldcommits) == 74
    rawmap = dict(csv.reader((RECORD / 'raw-rewritten-commits.csv').open()))
    assert set(rawmap) <= set(oldcommits)
    mapping = {old: rawmap.get(old, old) for old in oldcommits}
    assert len(set(mapping.values())) == 74
    newhead = git(PUB, 'rev-parse', 'HEAD').decode().strip()
    assert mapping[OLD] == newhead
    assert set(git(PUB, 'rev-list', newhead, '^' + BASE).decode().splitlines()) == set(mapping.values())
    attr_lines = {f'{p} filter=lfs diff=lfs merge=lfs -text' for p in ARTS}
    seen = set()
    for old, new in mapping.items():
        a = git(SOURCE, 'cat-file', 'commit', old)
        b = git(PUB, 'cat-file', 'commit', new)
        ah, am = a.split(b'\n\n', 1); bh, bm = b.split(b'\n\n', 1)
        assert am == bm
        al = ah.splitlines(); bl = bh.splitlines()
        assert [l for l in al if not l.startswith((b'tree ', b'parent '))] == [l for l in bl if not l.startswith((b'tree ', b'parent '))]
        parents = [l[7:].decode() for l in al if l.startswith(b'parent ')]
        assert [mapping.get(p, p) for p in parents] == [l[7:].decode() for l in bl if l.startswith(b'parent ')]
        at, bt = tree(SOURCE, old), tree(PUB, new)
        assert set(at) - set(ARTS) == (set(bt) - set(ARTS) - {'.gitattributes'})
        for p, info in at.items():
            if p in ARTS:
                assert info[2] == ARTS[p]['git_blob'] and bt[p][:2] == info[:2]
                pointer(PUB, bt[p][2], ARTS[p]); seen.add(p)
            else:
                assert bt[p] == info, (old, p)
        if '.gitattributes' in bt:
            assert bt['.gitattributes'][:2] == ['100644', 'blob']
            attrs = git(PUB, 'cat-file', 'blob', bt['.gitattributes'][2]).decode().splitlines()
            assert attrs and set(attrs) <= attr_lines and len(attrs) == len(set(attrs))
    assert seen == set(ARTS)
    with (RECORD / 'complete-commit-map.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['source_commit', 'publication_commit']); w.writerows(mapping.items())
    data = git(PUB, 'rev-list', '--objects', newhead)
    proc = subprocess.run(['git', 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'], input=data, cwd=PUB, env=ENV, stdout=subprocess.PIPE, check=True)
    blobs = [line.split() for line in proc.stdout.decode().splitlines() if ' blob ' in line]
    assert all(int(line[2]) < 104857600 for line in blobs)
    for p in ARTS:
        attrs = git(PUB, 'check-attr', 'filter', 'diff', 'merge', 'text', '--', p).decode().splitlines()
        assert [line.rsplit(': ',1)[1] for line in attrs] == ['lfs','lfs','lfs','unset']
    command('publisher-lfs-checkout', PUB, ['git', 'lfs', 'checkout'])
    for p,a in ARTS.items():
        assert filehash(PUB / p) == a['sha256'] and (PUB / p).stat().st_size == a['size_bytes']
    command('publisher-lfs-fsck', PUB, ['git', 'lfs', 'fsck', newhead])
    assert not git(PUB, 'status', '--porcelain').strip()
    save('migration-validation.json', {'status':'PASS', 'source_head':OLD, 'migrated_head':newhead, 'base':BASE, 'source_and_mapped_commits':74, 'rewritten_csv_rows':len(rawmap), 'identity_rows':74-len(rawmap), 'exact_metadata_messages_parents_and_unaffected_trees':True, 'historical_pointers_verified':sorted(seen), 'head_hydrated_sha256_verified':True, 'lfs_fsck_exit':0, 'ordinary_blob_count':len(blobs), 'largest_ordinary_blob_bytes':max(int(b[2]) for b in blobs), 'complete_map_sha256':filehash(RECORD / 'complete-commit-map.csv')})
    print('Migration validation PASS:', len(mapping), 'commits preserved; all four payloads exact', flush=True)

def remoteverify():
    assert not VERIFY.exists()
    expected = git(PUB, 'rev-parse', 'HEAD').decode().strip()
    skip = dict(ENV, GIT_LFS_SKIP_SMUDGE='1')
    command('fresh-github-clone', ROOT, ['git','clone','--single-branch','--branch',BRANCH,URL,str(VERIFY)], skip)
    assert git(VERIFY,'rev-parse','HEAD').decode().strip() == expected
    isolation(VERIFY)
    storage = VERIFY / '.git/independent-lfs-storage'
    assert not storage.exists()
    setup(VERIFY, storage)
    assert not storage.exists() or not any(p.is_file() for p in storage.rglob('*'))
    assert git(VERIFY, 'config', '--get', 'lfs.storage').decode().strip() == str(storage)
    assert git(VERIFY, 'config', '--get', 'lfs.url').decode().strip() == URL + '/info/lfs'
    assert not (VERIFY / '.lfsconfig').exists()
    save('remote-empty-cache.json', {'status':'PASS', 'empty_before_fetch':True, 'storage':str(storage),'origin':git(VERIFY,'remote','get-url','origin').decode().strip(),'endpoint':URL+'/info/lfs','expected_head':expected,'no_alternates':True})
    command('remote-lfs-fetch', VERIFY, ['git','lfs','fetch','origin','refs/heads/'+BRANCH])
    command('remote-lfs-checkout', VERIFY, ['git','lfs','checkout'])
    t = tree(VERIFY, expected)
    for p,a in ARTS.items():
        pointer(VERIFY,t[p][2],a)
        assert filehash(VERIFY/p) == a['sha256'] and (VERIFY/p).stat().st_size == a['size_bytes']
    for p in ['docs/team/PROJECT_HANDOFF.md','docs/team/CONTRIBUTIONS.md']:
        assert filehash(VERIFY/p) == filehash(SOURCE/p)
    command('remote-lfs-fsck', VERIFY, ['git','lfs','fsck',expected])
    assert not git(VERIFY,'status','--porcelain').strip()
    save('remote-download-verification.json', {'status':'PASS','publication_head':expected,'all_four_payloads_sha256_and_size_verified':True,'empty_independent_cache_before_github_fetch':True,'pointer_tree_identity_verified':True,'handoff_sha256':filehash(VERIFY/'docs/team/PROJECT_HANDOFF.md'),'ledger_sha256':filehash(VERIFY/'docs/team/CONTRIBUTIONS.md'),'private_origin':URL,'lfs_fsck_exit':0})
    print('Fresh GitHub download PASS: four files, handoff and ledger exact', flush=True)

def preserve():
    before = json.loads((RECORD/'source-preservation-baseline.json').read_text())
    after = snapshot()
    for key in before:
        if key not in ['log_size','log_sha256']:
            assert before[key] == after[key], key
    with (SOURCE/'reports/PAPER_EVIDENCE_LOG.md').open('rb') as f:
        assert digest(f.read(before['log_size'])) == before['log_sha256']
    save('source-preservation-final.json', {'status':'PASS','baseline':before,'final':after,'original_log_prefix_unchanged':True})
    print('Original source/PDF/index/config/refs and log-prefix preservation PASS',flush=True)

if __name__ == '__main__':
    action = sys.argv[1]
    try:
        globals()[action]()
    except Exception as e:
        save(action + '-failure.json', {'status':'FAIL','type':type(e).__name__,'reason':str(e)})
        raise
