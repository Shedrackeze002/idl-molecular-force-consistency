"""Frozen once-only supervisor with process-group timeout and no retry."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from train_offset_profile import cc,CONFIG_ID,verify_inputs,verify_gate

def main():
    config_path=Path('configs/aspirin-offset-profile.json');c=cc.read(config_path)
    if cc.canonical_hash(c)!=CONFIG_ID:raise ValueError('Unreviewed launch config')
    verify_inputs(c);gate=verify_gate(c);root=Path(c['prelaunch_gate']).parent
    owner=cc.DirectoryLease(root)
    try:
        owner.json('production-invocation.json',{'ordinal':1,'started_at_utc':cc.utc(),'config_sha256':cc.digest(config_path),
            'gate':gate,'run_directory':c['run_directory'],'no_retry':True},exclusive=True)
        if cc.no_symlinks(c['run_directory']).exists():raise FileExistsError('Production root already exists')
        start=time.perf_counter();command=[sys.executable,str(Path(__file__).parent/'train_offset_profile.py'),'--config',str(config_path)]
        with (root/'production-stdout.txt').open('xb') as out,(root/'production-stderr.txt').open('xb') as err:
            p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            owner.json('production-process.json',{'pid':p.pid,'command':command,'started_at_utc':cc.utc()})
            try:code=p.wait(timeout=c['supervisor_wall_seconds']);stop='returned'
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                stop='external_timeout_no_retry'
        owner.json('production-process-completion.json',{'exit_code':code,'stop_reason':stop,'elapsed_seconds':time.perf_counter()-start,
            'ended_at_utc':cc.utc(),'stdout_sha256':cc.digest(root/'production-stdout.txt'),'stderr_sha256':cc.digest(root/'production-stderr.txt')})
        return code
    finally:owner.close()
if __name__=='__main__':raise SystemExit(main())
