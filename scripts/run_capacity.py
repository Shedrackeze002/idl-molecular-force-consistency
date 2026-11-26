"""Once-only capacity launcher; preserves output and process failure without retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from force_consistency import capacity_contract as cc


def main():
    config_path = Path('configs/aspirin-capacity.json')
    c = cc.read(config_path);cc.validate_contract(c);cc.verify_sources(c)
    record_root = Path('reports/execution/capacity-2026-10-02')
    gate = cc.read(record_root / 'preproduction-gate.json')
    from force_consistency.runs import code_provenance
    code = code_provenance(Path.cwd());cc.verify_code(code)
    if (gate['status'] != 'cleared' or gate['config_sha256'] != cc.digest(config_path)
        or gate['code_fingerprint'] != code['code_fingerprint']
        or gate['launcher_sha256'] != cc.digest(__file__)
        or any(cc.digest(name) != sha for name, sha in gate['verification_artifacts'].items())):
        raise ValueError('Preproduction review/test/launcher identity changed')
    owner = cc.DirectoryLease(record_root)
    try:
        owner.json('production-invocation.json', {'ordinal': 1, 'started_at_utc': cc.utc(),
                   'configuration': str(config_path), 'config_sha256': cc.digest(config_path), 'code': code,
                   'run_directory': c['run_directory'], 'no_retry': True}, exclusive=True)
        cc.no_symlinks(c['run_directory'])
        if Path(c['run_directory']).exists():raise FileExistsError('Exclusive production root already exists')
        start = time.perf_counter();command = [sys.executable, '-m', 'force_consistency.capacity_runs', '--config', str(config_path)]
        with (record_root / 'production-stdout.txt').open('xb') as out, (record_root / 'production-stderr.txt').open('xb') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            owner.json('production-process.json', {'pid': process.pid, 'command': command, 'started_at_utc': cc.utc()})
            try:
                exit_code = process.wait(timeout=c['total_wall_seconds'] + 60)
                stop = 'returned'
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL);exit_code = process.wait()
                stop = 'external_operational_timeout_no_retry'
            owner.json('production-process-completion.json', {'exit_code': exit_code, 'stop_reason': stop,
                       'elapsed_seconds': time.perf_counter() - start, 'ended_at_utc': cc.utc(),
                       'stdout_sha256': cc.digest(record_root / 'production-stdout.txt'),
                       'stderr_sha256': cc.digest(record_root / 'production-stderr.txt')})
            return exit_code
    finally:owner.close()


if __name__ == '__main__':raise SystemExit(main())
