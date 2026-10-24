"""Small serial segment wrapper; checkpoint arithmetic stays in the frozen trainer."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from flax import serialization
from force_consistency.data.splits import canonical_hash
from force_consistency.runs import (_read_checkpoint,assert_finite,assert_parameter_dtype,
    atomic_write,code_provenance,load_run_control)
from force_consistency.train import RunConfig,train

ROOT=Path(__file__).resolve().parents[2]


class IdentityError(ValueError):
    """A different experiment must not become a corruption fallback."""


def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):atomic_write(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def runtime_identity(config):
    c=asdict(RunConfig(**config));m=read_json(Path(c['data_directory'])/'manifest.json')
    return {**code_provenance(ROOT),'dataset_identity':m['identity'],'split_hash':m['split_hash'],
        'policy_hash':m['policy_hash'],'unit_evidence_hash':m['unit_evidence_hash'],
        'config_computation_hash':canonical_hash({k:v for k,v in c.items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')}),
        'arm':c['arm'],'consistency_strength':c['consistency_strength'],'loss_weights':c['loss_weights'],
        'backend':'cpu','precision':'float64','normalization':m['normalization']}


def inspect_checkpoint(path,expected_identity):
    path=Path(path);prov=read_json(path.parent/'provenance.json')
    if any(prov.get(k)!=v for k,v in expected_identity.items()):raise IdentityError('Checkpoint source/config/data identity mismatch: '+str(path))
    envelope=_read_checkpoint(path,prov)
    state=serialization.msgpack_restore(envelope['payload']);control=load_run_control(path,prov)
    assert_parameter_dtype(state['params']);assert_finite(state)
    u=state['update'];history=control['validation_history'];best=control['best_update']
    if type(u) is not int or u<0 or u%500 or [h['update'] for h in history]!=list(range(0,u+1,500)):
        raise ValueError('Checkpoint validation schedule is ineligible')
    scores=[h['equal_connection_force_mae_eV_per_A'] for h in history]
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<0 for x in scores):raise ValueError('Invalid saved validation score')
    first=min(range(len(scores)),key=scores.__getitem__)
    if type(best) is not int or best!=history[first]['update'] or control['best_score']!=scores[first]:raise ValueError('BEST does not match saved schedule')
    selected=serialization.msgpack_restore(control['best_state']);assert_finite(selected);assert_parameter_dtype(selected['params'])
    if selected['update']!=best:raise ValueError('Embedded BEST update mismatch')
    return {'path':str(path),'update':u,'state':state,'control':control,'provenance':prov}


def newest_valid_checkpoint(segments,expected_identity):
    found=[];rejected=[]
    for segment in sorted(Path(segments).glob('[0-9]*')):
        for name in ('last.msgpack','scheduled.msgpack'):
            path=segment/name
            if not path.exists():continue
            try:found.append(inspect_checkpoint(path,expected_identity))
            except IdentityError:raise
            except Exception as exc:rejected.append({'path':str(path),'reason':f'{type(exc).__name__}: {exc}'})
    if not found:raise ValueError('No eligible scheduled checkpoint; refusing silent restart')
    return max(found,key=lambda x:(x['update'],x['path'])),rejected


def retain_scheduled(segment,expected_identity):
    segment=Path(segment);source=segment/'last.msgpack'
    if not source.exists() or not (segment/'provenance.json').exists():return False
    candidate=segment/'candidate.msgpack'
    try:
        atomic_write(candidate,source.read_bytes());inspect_checkpoint(candidate,expected_identity)
    except IdentityError:raise
    except Exception:return False
    os.replace(candidate,segment/'scheduled.msgpack');return True


def same_state(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(same_state(x,y) for x,y in zip(a,b))
    return bool(np.array_equal(a,b))


def completed_segment(segment,expected_identity,target=200000):
    segment=Path(segment);summary=read_json(segment/'summary.json')
    if summary.get('status')!='passed' or summary.get('stop_reason')!='max_updates' or summary.get('updates')!=target:
        raise ValueError('Fit is not complete')
    last=inspect_checkpoint(segment/'last.msgpack',expected_identity);best=inspect_checkpoint(segment/'best.msgpack',expected_identity)
    if not same_state(best['state'],serialization.msgpack_restore(last['control']['best_state'])):raise ValueError('BEST differs from authenticated selected state')
    if 'config_computation_hash' in expected_identity:
        cfg=asdict(RunConfig(**read_json(segment/'config.json')))
        ch=canonical_hash({k:v for k,v in cfg.items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')})
        if ch!=expected_identity['config_computation_hash']:raise IdentityError('Completed config mismatch')
    if last['update']!=target or summary.get('provenance')!=last['provenance'] or summary.get('best_update')!=last['control']['best_update'] or best['update']!=last['control']['best_update']:
        raise ValueError('Completed checkpoint/control mismatch')
    for role,update,name in [('last',target,'checkpoint-verification.json'),('best',best['update'],'best-checkpoint-verification.json')]:
        proof=read_json(segment/name)
        if proof.get('status')!='passed' or proof.get('role')!=role or proof.get('update')!=update:raise ValueError('Reload proof missing')
    return last


def run_fit(config,fit_root):
    fit_root=Path(fit_root);segments=fit_root/'segments';segments.mkdir(parents=True,exist_ok=True)
    expected=runtime_identity(config);existing=sorted(segments.glob('[0-9]*'));completion_rejected=[]
    for segment in reversed(existing):
        if not (segment/'summary.json').exists():continue
        try:
            summary=read_json(segment/'summary.json')
            if summary.get('status')=='passed' and summary.get('stop_reason')=='max_updates' and summary.get('updates')==config['max_updates']:
                complete=completed_segment(segment,expected,config['max_updates'])
                print('SKIP verified completed fit '+str(fit_root),flush=True)
                return {'status':'completed','segment':str(segment),'updates':complete['update'],'skipped':True}
        except IdentityError:raise
        except Exception as exc:completion_rejected.append({'path':str(segment),'reason':f'Completion rejected: {type(exc).__name__}: {exc}'})
    chosen,rejected=newest_valid_checkpoint(segments,expected) if existing else (None,[])
    if chosen and chosen['update']>config['max_updates']:raise ValueError('Resume state exceeds target')
    number=max([int(x.name) for x in existing],default=0)+1;segment=segments/f'{number:06d}';segment.mkdir(exist_ok=False)
    metadata=fit_root/'attempts'/f'{number:06d}';metadata.mkdir(parents=True,exist_ok=False)
    actual={**config,'run_directory':str(segment),'resume_checkpoint':chosen['path'] if chosen else None}
    write_json(metadata/'worker-config.json',actual);write_json(metadata/'resume-selection.json',{'checkpoint':chosen['path'] if chosen else None,'update':chosen['update'] if chosen else 0,'rejected':completion_rejected+rejected})
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+os.pathsep+str(ROOT/'src')
    log=metadata/'worker.log';print('START '+str(fit_root)+' at update '+str(chosen['update'] if chosen else 0),flush=True)
    started=time.perf_counter()
    with log.open('w') as output,log.open() as reader:
        proc=subprocess.Popen([sys.executable,'-m','experiments.representation_study.runner','--worker',str(metadata/'worker-config.json')],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        write_json(metadata/'process.json',{'pid':proc.pid,'started_unix_seconds':time.time(),'role':'training worker; metadata only, never an ownership gate'})
        stamp=None
        try:
            while proc.poll() is None:
                path=segment/'last.msgpack'
                if path.exists():
                    stat=path.stat();now=(stat.st_mtime_ns,stat.st_size)
                    if stamp!=now:retain_scheduled(segment,expected);stamp=now
                text=reader.read()
                if text:print(text,end='',flush=True)
                time.sleep(.2)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            retain_scheduled(segment,expected)
            text=reader.read()
            if text:print(text,end='',flush=True)
            write_json(metadata/'execution.json',{'elapsed_seconds':time.perf_counter()-started,'returncode':proc.returncode})
    summary=read_json(segment/'summary.json') if (segment/'summary.json').exists() else {'status':'interrupted','updates':None}
    if proc.returncode or summary.get('status')=='failed':raise RuntimeError('Training stopped; evidence retained in '+str(segment)+': '+str(summary.get('failure_reason',proc.returncode)))
    if summary.get('updates')!=config['max_updates'] or summary.get('stop_reason')!='max_updates':
        return {'status':'incomplete','segment':str(segment),'updates':summary.get('updates'),'stop_reason':summary.get('stop_reason')}
    completed_segment(segment,expected,config['max_updates'])
    return {'status':'completed','segment':str(segment),'updates':config['max_updates'],'skipped':False}


def verify_execution(settings):
    if settings.get('status')!='APPROVED_FOR_EXECUTION':raise IdentityError('Unreviewed launch record')
    if read_json(settings['parameter_counts_file'])['scalar_width']!=settings['scalar_width']:raise IdentityError('Scalar width differs from verified counts')
    if code_provenance(ROOT)['revision']!=settings['execution_revision']:raise IdentityError('Execution revision changed')
    for pin in settings.get('verification_receipts',[]):
        if hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()!=pin['sha256']:raise IdentityError('Verification receipt changed')
    for rel,digest in settings['implementation_sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise IdentityError('Implementation changed: '+rel)


def validate_references(settings,configs):
    references=settings['reference_runs']
    if set(references)!={'E64-0','E64-1'}:raise IdentityError('E64 reference labels must be seeds0/1')
    excluded={'data_directory','run_directory','wall_seconds','resume_checkpoint'}
    for label,path in references.items():
        actual=asdict(RunConfig(**read_json(Path(path)/'config.json')))
        if {k:v for k,v in actual.items() if k not in excluded}!={k:v for k,v in configs[label].items() if k not in excluded}:
            raise IdentityError('Historical reference label/config/seed mismatch: '+label)
    expected_data=read_json(Path(settings['data_directory'])/'manifest.json')['identity']
    for label,path in references.items():
        if read_json(Path(path)/'provenance.json')['dataset_identity']!=expected_data:raise IdentityError('Historical reference data mismatch')
        for name,digest in settings['reference_sha256'][label].items():
            if hashlib.sha256((Path(path)/name).read_bytes()).hexdigest()!=digest:raise IdentityError('Historical reference artifact changed: '+label+'/'+name)


def run_queue(settings):
    from .protocol import make_configs
    from .analysis import score_fit
    verify_execution(settings)
    configs=make_configs(settings['data_directory'],settings['output_root'],settings['scalar_width']);rows=[]
    validate_references(settings,configs)
    for key,run in settings['reference_runs'].items():
        print('SCORING reference '+key,flush=True)
        score=score_fit(run,Path(settings['output_root'])/'references'/key)
        rows.append({'candidate':key,'status':'completed','reused':True,'score':score})
    write_json(Path(settings['output_root'])/'queue.json',rows)
    order=[x for x in ('SP-0','S64-0','S64-1','SP-1') if x in configs]
    for key in order:
        verify_execution(settings);fit=Path(settings['output_root'])/key
        try:
            result=run_fit(configs[key],fit)
            if result['status']=='completed':result['score']=score_fit(result['segment'],fit/'analysis')
        except Exception as exc:
            result={'status':'failed','reason':f'{type(exc).__name__}: {exc}','fit_root':str(fit)}
        result['candidate']=key;rows.append(result);write_json(Path(settings['output_root'])/'queue.json',rows)
        from .analysis import paired_summary
        complete={x['candidate']:x for x in rows if x['status']=='completed'}
        paired=[{'candidate':k,'best':x['score']['roles']['best']['equal_connection_force_mae_eV_per_A'],'late20':x['score']['history']['late20_equal_connection_force_mae_eV_per_A']} for k,x in complete.items() if k.startswith('E64-') or all(f'{k.rsplit("-",1)[0]}-{seed}' in complete for seed in (0,1))]
        write_json(Path(settings['output_root'])/'paired.json',paired_summary(paired))
        if settings.get('completion_hook'):
            subprocess.run([*settings['completion_hook'],key,str(Path(settings['output_root'])/'queue.json')],check=True)
        if result['status']!='completed':break
    return rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--queue');args=parser.parse_args()
    if args.worker:
        cfg=read_json(args.worker);summary=train(RunConfig(**cfg));print(json.dumps({k:summary.get(k) for k in ('status','updates','stop_reason','elapsed_seconds','failure_reason')}),flush=True)
        raise SystemExit(1 if summary.get('status')=='failed' else 0)
    elif args.queue:
        rows=run_queue(read_json(args.queue));print(json.dumps(rows,indent=2))
        raise SystemExit(0 if all(x['status']=='completed' for x in rows) else 2)
    else:parser.error('--worker or --queue is required')
