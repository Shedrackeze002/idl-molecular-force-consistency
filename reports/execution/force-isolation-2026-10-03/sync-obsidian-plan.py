"""Retained planning-only navigation sync via authorized vault-first CLI."""
import hashlib
import json
import subprocess
from pathlib import Path

folder = Path(__file__).parent
chunk = (folder / 'obsidian-plan-text.md').read_text()
notes = [('hub', 'Project Hub.md'), ('findings', 'Research Findings.md')]
record = {'scope': 'planning_only_no_numerical_result', 'calls': [], 'notes': {}}

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def call(command, **options):
    number = len(record['calls']) + 1
    entry = {'number': number, 'command': command, 'path': options.get('path')}
    record['calls'].append(entry)
    try:
        result = subprocess.run(
            ['/usr/local/bin/obsidian', 'vault=hermes', command]
            + [key + '=' + value for key, value in options.items()],
            capture_output=True, text=True, timeout=20,
        )
        entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode or result.stdout.strip() in ('Vault not found.', 'File not found.') or result.stdout.startswith('Error:'):
            raise RuntimeError('CLI failed: ' + command)
        if command == 'read' and (len(result.stdout) < 100 or not result.stdout.lstrip().startswith('#')):
            raise RuntimeError('Actual project-note content not verified')
        return result.stdout
    except Exception as error:
        entry['error'] = str(error)
        raise
    finally:
        (folder / f'obsidian-plan-cli-call-{number}.json').write_text(json.dumps(entry, indent=2, sort_keys=True) + '\n')

try:
    before = {}
    # Both actual notes are backed up before the first mutation.
    for label, name in notes:
        value = call('read', path='Codex/IDL MLIP Project/' + name)
        before[label] = value
        with (folder / f'obsidian-{label}-before-plan.md').open('x') as stream:
            stream.write(value)
    for label, name in notes:
        path = 'Codex/IDL MLIP Project/' + name
        if chunk.strip() not in before[label]:
            escaped = chunk.replace('\\', '\\\\').replace('\n', '\\n')
            call('append', path=path, content=escaped)
        after = call('read', path=path)
        again = call('read', path=path)
        assert after == again, 'Exact repeated CLI readback differs'
        assert after.startswith(before[label].rstrip('\n')), 'Existing note prefix changed'
        if chunk.strip() not in before[label]:
            assert after[len(before[label].rstrip('\n')):].strip() == chunk.strip(), 'Exact appended text differs'
        else:
            assert after == before[label], 'Idempotent readback differs'
        with (folder / f'obsidian-{label}-plan.md').open('x') as stream:
            stream.write(after)
        record['notes'][label] = {'status': 'PASS_exact_CLI_readback', 'before_sha256': sha(before[label]), 'after_sha256': sha(after), 'backup_actual_note': True, 'prior_prefix_unchanged': True, 'exact_chunk_verified': True, 'repeated_readback_equal': True}
    record['status'] = 'PASS_both_navigation_notes'
except Exception as error:
    record['status'] = 'FAILED_operational_sync_preserved'
    record['error'] = str(error)
    raise
finally:
    with (folder / 'obsidian-plan-verification.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': record['status'], 'notes': record['notes'], 'calls': len(record['calls'])}))
