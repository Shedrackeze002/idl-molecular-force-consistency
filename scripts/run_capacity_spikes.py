"""Once-only supervisor for the reviewed frozen-checkpoint diagnostic."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from force_consistency import capacity_contract as cc


def main():
    path = Path('scripts/diagnose_capacity_spikes.py')
    spec = importlib.util.spec_from_file_location('capacity_spikes', path)
    analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)
    config_path = Path('configs/aspirin-capacity-spike.json'); config = cc.read(config_path)
    analysis.verify_gate(config); analysis.verify_inputs(config)
    records = Path('reports/execution/spike-diagnostic-2026-10-02')
    launch = cc.read(records / 'launch-guard.json')
    if (launch['launcher_sha256'] != cc.digest(__file__) or launch['config_sha256'] != cc.digest(config_path)
            or launch['prelaunch_gate_sha256'] != cc.digest(config['prelaunch_gate'])):
        raise ValueError('Frozen supervisor/config/gate differs')
    if Path(config['run_directory']).exists(): raise FileExistsError('Once-only numerical root exists')
    owner = cc.DirectoryLease(records)
    try:
        owner.json('production-invocation.json', {'ordinal': 1, 'no_retry': True, 'started_at_utc': cc.utc(),
                   'launcher_sha256': cc.digest(__file__), 'analysis_script_sha256': cc.digest(path),
                   'config_sha256': cc.digest(config_path), 'source_revision': launch['source_revision'],
                   'prelaunch_gate_sha256': cc.digest(config['prelaunch_gate'])}, exclusive=True)
        start = time.perf_counter(); command = [sys.executable, str(path), '--config', str(config_path)]
        with (records / 'production-stdout.txt').open('xb') as out, (records / 'production-stderr.txt').open('xb') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            owner.json('production-process.json', {'pid': process.pid, 'command': command, 'started_at_utc': cc.utc()})
            try:
                exit_code = process.wait(timeout=1860); stop = 'returned'
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); exit_code = process.wait()
                stop = 'external_operational_timeout_no_retry'
            owner.json('production-process-completion.json', {'exit_code': exit_code, 'stop_reason': stop,
                       'elapsed_seconds': time.perf_counter() - start, 'ended_at_utc': cc.utc(),
                       'stdout_sha256': cc.digest(records / 'production-stdout.txt'),
                       'stderr_sha256': cc.digest(records / 'production-stderr.txt')})
            return exit_code
    finally:
        owner.close()


if __name__ == '__main__':
    raise SystemExit(main())
