"""One guarded replacement matrix; pure child checks precede numerical imports."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import msgpack
from .width_rate_contract import read,digest,canonical_hash,resolve_commit,expected_updates,safe_path,MSE_KEYS
from .width_rate_replacement_contract import CONFIG_ID,PINS_ID,ROLE,ROOT,validate_replacement,verify_replacement_sources,no_symlinks,claim_root


def utc():return datetime.now(timezone.utc).isoformat()
def normalized(value):return json.loads(json.dumps(value,allow_nan=False))
def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def atomic(path,value):
    path=Path(path);no_symlinks(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(json_bytes(value));f.flush();os.fsync(f.fileno());temporary=f.name
    os.replace(temporary,path)


def exclusive(path,value):
    no_symlinks(path)
    with Path(path).open('xb') as f:f.write(json_bytes(value));f.flush();os.fsync(f.fileno())


def journal(root,phase,status,details,artifacts=()):
    path=Path(root)/'journal.jsonl';no_symlinks(path)
    row={'at_utc':utc(),'phase':phase,'status':status,'details':details,
         'artifacts_sha256':{str(p):digest(p) for p in artifacts if Path(p).is_file()}}
    with path.open('ab') as f:
        f.write(json.dumps(row,sort_keys=True,allow_nan=False).encode()+b'\n');f.flush();os.fsync(f.fileno())


def _engine():
    from . import width_rate_runs
    return width_rate_runs


def replacement_preflight(config,context,save_partial):
    sources=verify_replacement_sources(config)
    prepared=_engine().width_rate_preflight(sources['original_config'],context,save_partial)
    if prepared['sources']!=sources['original_sources']:raise ValueError('Original numerical preflight source facts changed')
    prepared['sources']=verify_replacement_sources(config)
    return prepared


def _audit_cell(config,root,sources):
    """Metadata-only complete-cell audit, also used directly by the public child."""
    validate_replacement(config);root=Path(root);no_symlinks(root)
    if read(root/'config.json')!=config:raise ValueError('Saved replacement config changed')
    case=read(root/'case.json');prov=read(root/'provenance.json');contract=read(root/'checkpoint-contract.json')
    if case not in config['cases'] or prov['case']!=case or contract['case']!=case:raise ValueError('Unreviewed replacement case')
    if (prov['config_identity']!=CONFIG_ID or contract['config_identity']!=CONFIG_ID
        or prov['source_pin_manifest_identity']!=PINS_ID or prov['sources']!=normalized(sources)
        or prov['normalization']!=sources['normalization'] or contract['normalization']!=sources['normalization']
        or prov['original_baselines']!=sources['original_baselines']
        or prov['model']!={**config['model'],'features':case['features']} or prov['backend']!='cpu' or prov['precision']!='float64'
        or prov['inputs']!=sources['historical_inputs'] or contract['inputs']!=sources['historical_inputs']):
        raise ValueError('Source-derived config/case/model/normalization/input provenance changed')
    for role in ('fitting','anchor'):
        path=root/f'{role}-inputs.npz';no_symlinks(path)
        source=Path(config['original_failed_run_directory'])/'width16-lr1e-3'/path.name
        expected_input=sources['pins'][str(source)]
        if digest(path)!=expected_input or prov['input_files_sha256'][role]!=expected_input or contract['input_files_sha256']!=prov['input_files_sha256']:
            raise ValueError('Source input physical bytes changed')
    receipt,history=resolve_commit(root)
    if receipt['update']!=6000 or contract['update']!=6000 or contract['commit']!=receipt or contract['state_hashes']!=receipt['state_hashes']:
        raise ValueError('Endpoint state/commit contract changed')
    if [r['update'] for r in history]!=expected_updates():raise ValueError('Complete64row history required')
    if contract['reference_sha256']!=digest(root/'reference.npz'):raise ValueError('Reference bytes changed')
    envelope=None
    for i,row in enumerate(history):
        update=row['update'];cp=root/f'snapshots/{update}.msgpack';hp=root/f'histories/{update}.json';no_symlinks(cp);no_symlinks(hp)
        if read(hp)!=history[:i+1]:raise ValueError('Committed history prefix changed')
        envelope=msgpack.unpackb(cp.read_bytes(),raw=False)
        if (set(envelope)!={'payload','sha256','provenance','control_payload','control_sha256'}
            or not isinstance(envelope['payload'],bytes) or hashlib.sha256(envelope['payload']).hexdigest()!=envelope['sha256']
            or not isinstance(envelope['control_payload'],bytes) or hashlib.sha256(envelope['control_payload']).hexdigest()!=envelope['control_sha256']
            or envelope['control_payload']!=b'\xc0'
            or envelope['provenance']!=json.dumps(prov,sort_keys=True,separators=(',',':'),allow_nan=False)):
            raise ValueError('Checkpoint envelope/payload/control/provenance changed')
        if row['frame_exposures']!=update*128:raise ValueError('History exposures changed')
        for role in ('fitting','anchor'):
            for key in MSE_KEYS:
                value=row['metrics'][role][key]
                if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError('History raw metric invalid')
                if update==0 and not math.isclose(value,sources['original_baselines'][role][key],rel_tol=1e-8,abs_tol=1e-10):raise ValueError('Original baseline changed')
                if case['name']=='width16-lr1e-3' and update in sources['historical'] and not math.isclose(value,sources['historical'][update][role][key],rel_tol=1e-8,abs_tol=1e-10):raise ValueError('Replay history changed')
    if receipt['checkpoint']!='snapshots/6000.msgpack' or receipt['history']!='histories/6000.json':raise ValueError('Endpoint paths changed')
    return {'case':case,'provenance':prov,'contract':contract,'commit':receipt,'history':history,
            'payload_sha256':envelope['sha256'],'control_sha256':envelope['control_sha256']}


def audit_replacement_cell(config,root):
    sources=verify_replacement_sources(config);root=Path(root);no_symlinks(root)
    if root.resolve()!=(Path.cwd()/ROOT/read(root/'case.json')['name']).resolve():raise ValueError('Unreviewed replacement cell root')
    value=_audit_cell(config,root,sources)
    parent=read(root.parent/'provenance.json')
    for key in ('revision','code_files','code_fingerprint','new_code_sha256'):
        if value['provenance'][key]!=parent[key]:raise ValueError('Cell production code provenance changed')
    if canonical_hash(parent['code_files'])!=parent['code_fingerprint']:raise ValueError('Code fingerprint changed')
    for names in (parent['code_files'],parent['new_code_sha256']):
        for name,want in names.items():
            no_symlinks(Path.cwd()/name)
            if digest(safe_path(Path.cwd(),name))!=want:raise ValueError('Executed code source changed')
    if set(parent['new_code_sha256'])!={'src/force_consistency/width_rate_replacement.py','src/force_consistency/width_rate_replacement_contract.py','scripts/report_width_rate_replacement.py'}:
        raise ValueError('New executed code identity incomplete')
    return value


def attempt_binding(root,audit):
    return {'role':ROLE,'ordinal':1,'config_identity':CONFIG_ID,'pin_manifest_identity':PINS_ID,
            'root':str(Path(root).resolve()),'case':audit['case'],'commit':audit['commit'],
            'reference_sha256':audit['contract']['reference_sha256'],'payload_sha256':audit['payload_sha256'],
            'control_sha256':audit['control_sha256'],'revision':audit['provenance']['revision']}


def _claim_attempt(root,sha,binding):
    path=Path(root)/'probe-attempt.json';no_symlinks(path)
    if digest(path)!=sha or read(path)!=binding:raise ValueError('Missing or mismatched probe reservation')
    exclusive(Path(root)/'probe-entry.json',{'attempt_sha256':sha,'binding':binding,'entered_at_utc':utc()})


def replacement_probe(root,attempt_sha256):
    root=Path(root);config=read(root/'config.json')
    audit=audit_replacement_cell(config,root)
    _claim_attempt(root,attempt_sha256,attempt_binding(root,audit))
    engine=_engine();engine.require_backend('cpu')
    result=engine._probe_saved(root)
    valid_child(audit,result)
    atomic(root/'probe-completion.json',{'status':'passed','attempt_sha256':attempt_sha256,
        'verification_sha256':digest(root/'checkpoint-verification.json'),'ended_at_utc':utc()})
    return result


def valid_child(audit,result):
    receipt=audit['commit'];prov=audit['provenance']
    expected={'status':'passed','update':6000,'adam_count':6000,'normalization':prov['normalization'],
        'state_hashes':receipt['state_hashes'],'checkpoint_sha256':receipt['checkpoint_sha256'],'history_sha256':receipt['history_sha256'],
        'case':audit['case'],'config_identity':CONFIG_ID,'fitting_record_count':128,'anchor_record_count':16,'precision':'float64',
        'template_initializations':1,'template_initializer_frames':16,'prediction_frames':144,'tolerance':1e-10}
    error=result.get('maximum_absolute_error')
    if (any(result.get(k)!=v for k,v in expected.items()) or not isinstance(error,(int,float)) or isinstance(error,bool)
        or not math.isfinite(error) or not 0<=error<=1e-10):raise ValueError('Strict fresh child receipt mismatch')
    return {'template_initializations':1,'template_initializer_frames':16,'prediction_frames':144,'training_updates':0}


def positive_allowance(value):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:raise TimeoutError('No positive verifier allowance')
    return value


def _launch(root,timeout,attempt_sha256,*,context=None):
    positive_allowance(timeout);root=Path(root)
    if context is not None:timeout=min(timeout,positive_allowance(context.remaining_seconds()))
    try:
        result=subprocess.run([sys.executable,'-m','force_consistency.width_rate_replacement','--probe',str(root.resolve()),'--attempt-sha256',attempt_sha256],
                              env={**os.environ,'JAX_PLATFORMS':'cpu'},capture_output=True,check=True,timeout=timeout)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
        for name,value in [('stdout',exc.output),('stderr',exc.stderr)]:
            if value is not None:
                try:(root/f'probe-{name}.txt').write_bytes(value.encode() if isinstance(value,str) else value)
                except Exception as error:exc.add_note('Log persistence failed: '+str(error))
        raise
    (root/'probe-stdout.txt').write_bytes(result.stdout);(root/'probe-stderr.txt').write_bytes(result.stderr)
    return read(root/'checkpoint-verification.json')


def launch_probe(root,context):
    context.check_budget();positive_allowance(context.remaining_seconds());root=Path(root);config=read(root/'config.json')
    audit=audit_replacement_cell(config,root);context.check_budget();positive_allowance(context.remaining_seconds())
    exclusive(root/'probe-attempt.json',attempt_binding(root,audit))
    sha=digest(root/'probe-attempt.json');context.check_budget();allowance=positive_allowance(context.remaining_seconds())
    result=_launch(root,allowance,sha,context=context);valid_child(audit,result)
    if read(root/'checkpoint-verification.json')!=result:raise ValueError('Saved child result changed')
    return result


def recover_child_work(config,root,sources):
    audit=_audit_cell(config,root,sources);result=read(Path(root)/'checkpoint-verification.json')
    return {'work':valid_child(audit,result),'receipt':result}


def run_replacement_arm(config,case,prepared,root,provenance,context):
    sources=verify_replacement_sources(config)
    if case not in config['cases']:raise ValueError('Unreviewed replacement cell')
    if Path(root).resolve()!=(Path.cwd()/ROOT/case['name']).resolve():raise ValueError('Unreviewed arm root')
    engine=_engine();from dataclasses import asdict
    from .data.prepare import array_identity
    fitting=prepared['fitting'];anchor=prepared['anchor'];norm=prepared['normalization'];entry=prepared['widths'][case['features']]
    inputs={r:{'arrays':{k:array_identity(v) for k,v in engine._arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships}
            for r,p in [('fitting',fitting),('anchor',anchor)]}
    if (inputs!=sources['historical_inputs'] or asdict(norm)!=sources['normalization'] or prepared['inputs']!=inputs
        or prepared['sources']!=sources or asdict(entry['model'].config)!={**config['model'],'features':case['features']}):
        raise ValueError('Actual prepared replacement inputs/model/source facts changed')
    historical=sources['historical'] if case['name']=='width16-lr1e-3' else {}
    engine.validate_state(entry['state'],norm,0)
    out=engine._run_arm(config,case,entry['state'],fitting,anchor,entry['model'],norm,sources['original_baselines'],historical,root,provenance,context,probe_runner=launch_probe)
    try:
        verify_replacement_sources(config);context.check_budget();out['source_hashes_unchanged']=True
        if out['execution_valid']:valid_child(audit_replacement_cell(config,root),out['checkpoint_verification'])
    except BaseException as exc:out.update(status='failed',execution_valid=False,failure_reason=f'Post-cell guard: {type(exc).__name__}: {exc}')
    # Valid receipted child work remains known even when a later boundary fails.
    try:
        known=recover_child_work(config,root,sources)
        out.update(known_child_work=known['work'],known_child_receipt=known['receipt'],probe_work_status='completed')
    except (OSError,ValueError,KeyError,TypeError):
        out['known_child_work']=None if (Path(root)/'probe-attempt.json').exists() else {'template_initializations':0,'template_initializer_frames':0,'prediction_frames':0,'training_updates':0}
    out['elapsed_seconds']=context.clock()-context.arm_start;out['charged_elapsed_seconds']=out['elapsed_seconds']+context.shared_initialization_seconds
    try:atomic(Path(root)/'summary.json',out);context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,final_persistence_failure=f'{type(exc).__name__}: {exc}')
        atomic(Path(root)/'failure.json',out)
    return out


def evidence(config,identity,title,details):
    with Path(config['evidence_log']).open('a') as f:
        f.write('\n## '+identity+' — '+title+'\n\n```json\n'+json.dumps(details,sort_keys=True,indent=2,allow_nan=False)+'\n```\n');f.flush();os.fsync(f.fileno())



def failure_outcome(config,root,error,records):
    """Controller failure cannot erase earlier durable cells or invent their work."""
    root=Path(root);arms={};unreadable=[]
    for case in config['cases']:
        child=root/case['name'];arm=None
        for name in ('failure.json','summary.json'):
            path=child/name
            if path.is_file():
                try:arm=read(path);break
                except (OSError,ValueError) as exc:unreadable.append({'path':str(path),'sha256':digest(path),'error':str(exc)})
        if arm is None:
            arm={'case':case,'status':'unavailable' if child.exists() else 'blocked','executed':None if child.exists() else False,
                 'execution_valid':False,'stop_reason':'controller_failure_with_unreceipted_work' if child.exists() else 'preflight_or_controller_failure'}
        arms[case['name']]=arm
    return {'status':'failed','execution_valid':False,'full_pilot':'pending','failure_reason':f'{type(error).__name__}: {error}',
            'preflight_work':records,'arms':arms,'unreadable_terminal_artifacts':unreadable}


def diagnose_width_rate_replacement(config):
    sources=verify_replacement_sources(config);root=Path(config['run_directory'])
    # Root reservation is the matrix invocation; no failure writes into an unowned root.
    root=claim_root(root);owned=(root.stat().st_dev,root.stat().st_ino)
    global_start=time.perf_counter()
    exclusive(root/'invocation.json',{'role':ROLE,'ordinal':1,'config_identity':CONFIG_ID,'started_at_utc':utc()})
    evidence(config,'WIDTHRATE-REPLACEMENT-PRE-001','one fresh matrix starts',{'config':config,'config_identity':CONFIG_ID,'pin_identity':PINS_ID,
       'rationale':'Fresh equal-exposure width/fixed-rate sensitivity; original failed cell and copied verification remain separate.',
       'acceptance':'Original raw joint E/F<.1 baseline on fitting and fully fitted anchors; pilot pending, no automatic retry.'})
    engine=_engine();context=engine.ExecutionContext(config['total_wall_seconds'],global_start=global_start);records=[];started=utc()
    def partial(record):
        records.append(copy.deepcopy(record));atomic(root/'preflight-work.json',records)
        journal(root,'preflight',record['status'],record,[root/'preflight-work.json'])
    try:
        atomic(root/'config.json',config);journal(root,'matrix','started',{'config_identity':CONFIG_ID})
        prepared=replacement_preflight(config,context,partial);sources=prepared['sources'];atomic(root/'sources.json',sources)
        provenance={**engine.code_provenance(Path.cwd()),'prepared_identity':config['expected_prepared_identity'],'split_hash':config['expected_split_hash'],
                    'subsets_identity':config['expected_subsets_identity'],'source_pin_manifest_identity':PINS_ID,'sources':sources,
                    'new_code_sha256':{n:digest(n) for n in ('src/force_consistency/width_rate_replacement.py','src/force_consistency/width_rate_replacement_contract.py','scripts/report_width_rate_replacement.py')}}
        atomic(root/'provenance.json',provenance)
        from dataclasses import asdict
        for width,entry in prepared['widths'].items():
            engine.save_checkpoint(root/f'initial-width{width}.msgpack',entry['state'],{'features':width,'normalization':asdict(prepared['normalization']),'state_hashes':engine.canonical_state_hashes(entry['state'])})
        def runner(case):
            entry=prepared['widths'][case['features']];armcontext=context.for_arm(case['wall_seconds'],entry['initialization_seconds'])
            journal(root,case['name'],'started',{'case':case,'initial_state_hashes':engine.canonical_state_hashes(entry['state']),'fresh_count':0})
            evidence(config,'WIDTHRATE-REPLACEMENT-'+case['name']+'-PRE','fresh cell starts',{'case':case,'fresh_count':0,'initial_state_hashes':engine.canonical_state_hashes(entry['state'])})
            arm=run_replacement_arm(config,case,prepared,root/case['name'],provenance,armcontext)
            files=[p for p in (root/case['name']).rglob('*') if p.is_file()]
            detail={'case':case,'status':arm['status'],'execution_valid':arm['execution_valid'],'updates':arm.get('updates'),
                'persisted_update':arm.get('persisted_update'),'attempted_updates':arm.get('attempted_updates'),'pending_update_unknown':arm.get('pending_update_unknown'),
                'failure_reason':arm.get('failure_reason'),'stop_reason':arm.get('stop_reason'),'fit_gate':arm.get('fit_gate'),'anchor_gate':arm.get('anchor_gate'),
                'known_child_work':arm.get('known_child_work'),'raw_files_sha256':{str(p):digest(p) for p in files}}
            journal(root,case['name'],arm['status'],detail,files)
            evidence(config,'WIDTHRATE-REPLACEMENT-'+case['name']+'-RESULT','every cell result retained',detail)
            return arm
        out=engine.execute_width_rate_matrix(root,config,runner,lambda:verify_replacement_sources(config),context)
        out['parent_work']=prepared['parent_work'];out['shared_initializations']={str(k):v['receipt'] for k,v in prepared['widths'].items()}
    except BaseException as exc:
        out=failure_outcome(config,root,exc,records)
    out.update(role=ROLE,config_identity=CONFIG_ID,started_at_utc=started,ended_at_utc=utc(),elapsed_seconds=context.clock()-context.global_start)
    def owns():
        return not root.is_symlink() and root.exists() and (root.stat().st_dev,root.stat().st_ino)==owned
    try:
        if not owns():raise ValueError('Production root ownership lost')
        for name,arm in out['arms'].items():
            if arm['status']=='blocked':
                journal(root,name,'blocked',arm)
                evidence(config,'WIDTHRATE-REPLACEMENT-'+name+'-RESULT','blocked cell preserved',arm)
        context.check_budget();atomic(root/'summary.json',out);context.check_budget()
        atomic(root/'controller-completion.json',{'status':out['status'],'summary_sha256':digest(root/'summary.json'),'elapsed_seconds':context.clock()-context.global_start,'ended_at_utc':utc()});context.check_budget()
    except BaseException as exc:
        out.update(status='failed',execution_valid=False,final_output_failure=f'{type(exc).__name__}: {exc}',elapsed_seconds=context.clock()-context.global_start)
        if owns():atomic(root/'controller-failure.json',out)
        else:
            with tempfile.NamedTemporaryFile(prefix='width-rate-replacement-unowned-',suffix='.json',delete=False) as f:
                out['external_failure_record']=f.name;f.write(json_bytes(out));f.flush();os.fsync(f.fileno())
    if owns():journal(root,'matrix',out['status'],out)
    evidence(config,'WIDTHRATE-REPLACEMENT-RESULT-001','matrix terminal state and all outcomes',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--config',type=Path);choice.add_argument('--probe',type=Path);parser.add_argument('--attempt-sha256')
    args=parser.parse_args()
    if args.probe:
        if not args.attempt_sha256:parser.error('--probe requires --attempt-sha256')
        result=replacement_probe(args.probe,args.attempt_sha256);print(json.dumps(result));raise SystemExit(0)
    config=read(args.config);validate_replacement(config);result=diagnose_width_rate_replacement(config)
    print(json.dumps({k:result.get(k) for k in ('status','execution_valid','elapsed_seconds','failure_reason')}));raise SystemExit(0 if result['execution_valid'] else 1)
