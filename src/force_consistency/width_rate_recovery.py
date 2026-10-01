"""Opaque checkpoint-copy audit and one bounded fresh verifier; no numerical imports."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import msgpack
from .width_rate_contract import canonical_hash,digest,read,safe_path
from .width_rate_recovery_contract import CONFIG_ID,ROLE,validate,verify_recovery_sources


def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(data);f.flush();os.fsync(f.fileno());temporary=f.name
    os.replace(temporary,path)


def _expected(v):
    source=v['source_root'];c=v['config'];receipt=v['source_receipt'];prov=v['provenance'];contract=v['source_contract']
    original=(safe_path(source,receipt['checkpoint'])).read_bytes()
    if hashlib.sha256(original).hexdigest()!=receipt['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    envelope=msgpack.unpackb(original,raw=False)
    if set(envelope)!={'payload','sha256','provenance','control_payload','control_sha256'}:raise ValueError('Envelope fields changed')
    for field,key,expected in [('payload','sha256','expected_source_payload_sha256'),('control_payload','control_sha256','expected_source_control_sha256')]:
        if not isinstance(envelope[field],bytes) or hashlib.sha256(envelope[field]).hexdigest()!=envelope[key] or envelope[key]!=c[expected]:raise ValueError('Opaque state/control changed')
    if json.loads(envelope['provenance'])!=prov:raise ValueError('Semantic provenance changed')
    envelope['provenance']=json.dumps(prov,sort_keys=True,separators=(',',':'),allow_nan=False)
    checkpoint=msgpack.packb(envelope,use_bin_type=True);new_receipt={**receipt,'checkpoint_sha256':hashlib.sha256(checkpoint).hexdigest()}
    new_contract={**contract,'commit':new_receipt}
    names=['config.json','case.json','provenance.json','fitting-inputs.npz','anchor-inputs.npz','reference.npz',receipt['history']]
    data={name:safe_path(source,name).read_bytes() for name in names}
    if hashlib.sha256(data[receipt['history']]).hexdigest()!=receipt['history_sha256']:raise ValueError('Source history changed')
    data.update({receipt['checkpoint']:checkpoint,'commit.json':json_bytes(new_receipt),'checkpoint-contract.json':json_bytes(new_contract)})
    for name in data:
        if 'source_files_sha256' in v and digest(safe_path(source,name))!=v['source_files_sha256'][name]:raise ValueError('Source swapped after guard: '+name)
    transformation={'status':'completed','role':ROLE,'config_identity':canonical_hash(c),'source_root':str(source),
        'source_checkpoint_sha256':receipt['checkpoint_sha256'],'copied_checkpoint_sha256':new_receipt['checkpoint_sha256'],
        'source_history_sha256':receipt['history_sha256'],'payload_sha256':envelope['sha256'],'control_sha256':envelope['control_sha256'],
        'source_files_sha256':{name:digest(safe_path(source,name)) for name in data},
        'copied_files_sha256':{name:hashlib.sha256(b).hexdigest() for name,b in data.items()},'allowed_changes':['envelope.provenance','commit.checkpoint_sha256','contract.commit.checkpoint_sha256']}
    return data,transformation


def _audit_copy(v,root,transformation,allow_probe_receipt=False):
    root=Path(root);child=root/'verification-copy';data,expected=_expected(v)
    if transformation!=expected:raise ValueError('Transformation receipt changed')
    if child.is_symlink() or root.is_symlink():raise ValueError('Symlinked copy root')
    files={str(p.relative_to(child)) for p in child.rglob('*') if p.is_file() or p.is_symlink()}
    permitted=set(data)|({'checkpoint-verification.json'} if allow_probe_receipt else set())
    if not set(data)<=files or not files<=permitted:raise ValueError('Staged file set incomplete or substituted')
    for name,wanted in data.items():
        path=child/name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=child.parent) or safe_path(child,name).read_bytes()!=wanted:raise ValueError('Staged bytes/deltas changed: '+name)
    return expected


def _stage_copy(v,root,check_budget):
    root=Path(root);child=root/'verification-copy'
    if child.exists() or child.is_symlink():raise ValueError('Protected copied artifact already exists')
    check_budget();data,t=_expected(v);child.mkdir(parents=True)
    for name,b in data.items():
        if name=='commit.json':continue
        check_budget();atomic(safe_path(child,name),b)
    check_budget();atomic(child/'commit.json',data['commit.json'])
    # Candidate audit first, completed receipt publication second, published audit third.
    _audit_copy(v,root,t);check_budget();atomic(root/'copy-transformation.json',json_bytes(t))
    _audit_copy(v,root,read(root/'copy-transformation.json'));return t


def stage_recovery_copy(config,root,check_budget):
    v=verify_recovery_sources(config)
    if Path(root).resolve()!=Path(config['run_directory']).resolve() or Path(root).exists() or Path(root).is_symlink():raise ValueError('Unreviewed/existing output root')
    return _stage_copy(v,root,check_budget)


def audit_recovery_copy(config,root,*,allow_probe_receipt=False):
    v=verify_recovery_sources(config)
    if Path(root).resolve()!=Path(config['run_directory']).resolve():raise ValueError('Unreviewed audit root')
    return _audit_copy(v,root,read(Path(root)/'copy-transformation.json'),allow_probe_receipt)


def original_effort(v):
    report=v['old_report']
    return {'actual_counts':copy.deepcopy(report['actual_counts']),'shared_initializations':copy.deepcopy(report.get('shared_initializations',{})),
            'per_arm':{name:{k:copy.deepcopy(value) for k,value in arm.items() if any(s in k for s in ('frame','update','seconds','work_status','evaluation_pass','oracle_verification'))} for name,arm in report['arms'].items()}}


def original_outcome(v):
    report=v['old_report']
    return {k:copy.deepcopy(report[k]) for k in ('status','execution_valid','interpretation_valid','full_pilot')}|{'arms':{n:{k:copy.deepcopy(a.get(k)) for k in ('status','fit_gate','anchor_gate')} for n,a in report['arms'].items()}}


def code_record():
    names=['src/force_consistency/width_rate_recovery.py','src/force_consistency/width_rate_recovery_contract.py','src/force_consistency/width_rate_runs.py','src/force_consistency/width_rate_contract.py','uv.lock']
    return {'files_sha256':{n:digest(n) for n in names if Path(n).exists()},'python':platform.python_version(),'executable':sys.executable,
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'dependencies':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','e3x','numpy','msgpack')}}


def valid_child(v,root,result):
    c=v['config'];receipt=read(Path(root)/'verification-copy/commit.json');case=read(v['source_root']/'case.json');config=read(v['source_root']/'config.json')
    expected={'status':'passed','update':c['expected_update'],'adam_count':c['expected_update'],'normalization':v['provenance']['normalization'],'state_hashes':receipt['state_hashes'],
        'checkpoint_sha256':receipt['checkpoint_sha256'],'history_sha256':receipt['history_sha256'],'case':case,'config_identity':canonical_hash(config),
        'fitting_record_count':c['fitting_frame_count'],'anchor_record_count':c['anchor_frame_count'],'tolerance':c['checkpoint_prediction_atol'],'precision':'float64',
        'template_initializations':1,'template_initializer_frames':min(config['microbatch_size'],c['anchor_frame_count']),'prediction_frames':c['fitting_frame_count']+c['anchor_frame_count']}
    value=result.get('maximum_absolute_error')
    if any(result.get(k)!=val for k,val in expected.items()) or not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=c['checkpoint_prediction_atol']:raise ValueError('Child receipt mismatch')
    if result!=read(Path(root)/'verification-copy/checkpoint-verification.json'):raise ValueError('Saved child receipt changed')
    return {'training_updates':0,'optimizer_proposals':0,'warmup_frames':0,'template_initializations':1,'template_initializer_frames':expected['template_initializer_frames'],'prediction_frames':expected['prediction_frames']}


def _launch(root,timeout):
    env={**os.environ,'JAX_PLATFORMS':'cpu'}
    completed=subprocess.run([sys.executable,'-m','force_consistency.width_rate_runs','--probe',str((Path(root)/'verification-copy').resolve())],env=env,capture_output=True,timeout=timeout,check=True)
    atomic(Path(root)/'probe-stdout.txt',completed.stdout);atomic(Path(root)/'probe-stderr.txt',completed.stderr)
    return read(Path(root)/'verification-copy/checkpoint-verification.json')


def _recover(config,root,guard,probe_runner,*,clock=time.perf_counter):
    root=Path(root)
    if root.exists() or root.is_symlink():raise ValueError('Protected recovery root already exists')
    start=clock();deadline=start+config['total_wall_seconds'];probe_start=None;v=None
    def check():
        if clock()>=deadline:raise TimeoutError('Recovery boundary cap exhausted')
    out={'role':ROLE,'status':'failed','verification_valid':False,'config_identity':canonical_hash(config),'config':config,'full_pilot':'pending',
         'probe_attempts':0,'probe_work_status':'not_started','probe_seconds':0.,'new_work':{'training_updates':0,'optimizer_proposals':0,'warmup_frames':0,'template_initializations':0,'template_initializer_frames':0,'prediction_frames':0}}
    try:
        check();v=guard();check();root.mkdir(parents=True);out.update(original_effort=original_effort(v),original_outcome=original_outcome(v),code=code_record())
        atomic(root/'config.json',json_bytes(config));atomic(root/'prospective.json',json_bytes(out))
        _stage_copy(v,root,check);v=guard();_audit_copy(v,root,read(root/'copy-transformation.json'));check()
        out.update(probe_attempts=1,probe_work_status='started');atomic(root/'probe-work.json',json_bytes({'status':'started','phase':'launch_intent','attempt':1}));check();probe_start=clock()
        result=probe_runner(root,deadline-probe_start);out['probe_seconds']=clock()-probe_start;out['child_receipt']=result
        out['new_work']=valid_child(v,root,result);out['probe_work_status']='completed';atomic(root/'probe-work.json',json_bytes({'status':'completed','attempt':1,'new_work':out['new_work'],'elapsed_seconds':out['probe_seconds']}))
        check();v=guard();_audit_copy(v,root,read(root/'copy-transformation.json'),True);check();out.update(status='passed',verification_valid=True,elapsed_seconds=clock()-start)
        atomic(root/'summary.json',json_bytes(out));check()
        atomic(root/'controller-completion.json',json_bytes({'status':'passed','summary_sha256':digest(root/'summary.json'),'elapsed_seconds':clock()-start}));check()
    except BaseException as exc:
        if not isinstance(exc,(Exception,KeyboardInterrupt,SystemExit)):raise
        if out['probe_work_status']=='started':
            if probe_start is None:out.update(probe_work_status='not_started',probe_attempts=0)
            else:
                out.update(probe_work_status='unknown',probe_seconds=clock()-probe_start)
                out['new_work'].update(template_initializations=None,template_initializer_frames=None,prediction_frames=None)
        out.update(status='incomplete' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'failed',verification_valid=False,failure_reason=f'{type(exc).__name__}: {exc}',elapsed_seconds=clock()-start)
        if isinstance(exc,subprocess.CalledProcessError):out['probe_stderr']=(exc.stderr.decode(errors='replace') if isinstance(exc.stderr,bytes) else str(exc.stderr))[-4000:]
        if root.exists():
            try:atomic(root/'controller-failure.json',json_bytes(out))
            except Exception as persistence:out['failure_record_error']=str(persistence)
    return out


def recover_width_rate(config):
    validate(config)
    with Path(config['evidence_log']).open('a') as f:f.write('\n## WIDTHRATE-RECOVERY-PRE-001 — one copied checkpoint verification\n\nZero training; one180s child attempt; original failed matrix preserved. Config '+CONFIG_ID+'; original checkpoint '+config['expected_source_checkpoint_sha256']+'.\n')
    out=_recover(config,Path(config['run_directory']),lambda:verify_recovery_sources(config),_launch)
    with Path(config['evidence_log']).open('a') as f:f.write('\n## WIDTHRATE-RECOVERY-RESULT-001 — separate verifier outcome retained\n\n```json\n'+json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n```\n')
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args();out=recover_width_rate(read(args.config));print(json.dumps({k:out.get(k) for k in ('status','verification_valid','probe_attempts','elapsed_seconds','failure_reason')},indent=2));raise SystemExit(0 if out['verification_valid'] else 1)
