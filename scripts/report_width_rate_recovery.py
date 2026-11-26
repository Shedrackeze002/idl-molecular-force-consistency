"""Separate saved-byte recovery validation, never retroactive matrix clearance."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from force_consistency.width_rate_contract import canonical_hash,digest,read
from force_consistency.width_rate_recovery_contract import ROLE,verify_recovery_sources
from force_consistency.width_rate_recovery import _audit_copy,valid_child,original_effort,original_outcome,code_record


def _work(unknown=False):
    return {'training_updates':0,'optimizer_proposals':0,'warmup_frames':0,
            'template_initializations':None if unknown else 0,'template_initializer_frames':None if unknown else 0,'prediction_frames':None if unknown else 0}


def _number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


def _accounting(v,root,summary):
    elapsed=summary['elapsed_seconds'];probe=summary['probe_seconds'];attempt=summary['probe_attempts'];status=summary['probe_work_status']
    if not _number(elapsed) or not _number(probe) or probe>elapsed or type(attempt) is not int or attempt not in (0,1):raise ValueError('Invalid controller/probe timing or attempts')
    if status=='not_started':
        if attempt!=0 or probe!=0 or summary['new_work']!=_work():raise ValueError('Not-started work mismatch')
    elif status=='unknown':
        receipt=read(root/'probe-work.json')
        if attempt!=1 or summary['new_work']!=_work(True) or receipt.get('status')!='started' or receipt.get('attempt')!=1:raise ValueError('Unknown child work mismatch')
        if (root/'verification-copy/checkpoint-verification.json').exists():raise ValueError('Unknown work has an unaccounted child receipt')
    elif status=='completed':
        work=valid_child(v,root,summary['child_receipt']);receipt=read(root/'probe-work.json')
        if attempt!=1 or summary['new_work']!=work or receipt!={'status':'completed','attempt':1,'new_work':work,'elapsed_seconds':probe}:raise ValueError('Completed child work mismatch')
    else:raise ValueError('Unrecognized child work status')


def _available_work(v,root):
    # Interrupted records cannot establish issuance from launch intent alone.
    evidence=(root/'probe-work.json').exists() or (root/'verification-copy/checkpoint-verification.json').exists()
    result={'probe_attempts':None if evidence else 0,'probe_work_status':'unknown' if evidence else 'not_started','new_work':_work(evidence),'probe_seconds':None if evidence else 0.}
    try:
        receipt=read(root/'verification-copy/checkpoint-verification.json');work=valid_child(v,root,receipt)
        _audit_copy(v,root,read(root/'copy-transformation.json'),True)
        result.update(probe_attempts=1,probe_work_status='completed',child_receipt=receipt,new_work=work)
    except (OSError,ValueError,KeyError,TypeError):pass
    return result


def _invalid_json_constant(value):raise ValueError("Nonfinite terminal JSON: "+value)


def _generate(v,root,output):
    root=Path(root);output=Path(output);errors=[];accounting=False
    file=root/'controller-failure.json' if (root/'controller-failure.json').exists() else root/'summary.json'
    try:
        summary=json.loads(file.read_text(),parse_constant=_invalid_json_constant)
        if not isinstance(summary,dict):raise ValueError('Terminal record is not an object')
    except (OSError,ValueError,TypeError) as exc:
        errors.append('Missing/corrupt terminal record: '+str(exc));summary={'status':'incomplete','verification_valid':False,'elapsed_seconds':None,**_available_work(v,root)}
    valid=summary.get('verification_valid') is True and summary.get('status')=='passed'
    if not errors:
        try:
            if summary['role']!=ROLE or summary['config']!=v['config'] or summary['config_identity']!=canonical_hash(v['config']) or summary['full_pilot']!='pending' or summary['status'] not in ('passed','failed','incomplete'):raise ValueError('Recovery controls/role changed')
            if summary['original_effort']!=original_effort(v) or summary['original_outcome']!=original_outcome(v):raise ValueError('Original failed outcome/effort changed')
            if summary['code']['files_sha256']!=code_record()['files_sha256']:raise ValueError('Executed numerical/audit code changed')
            _audit_copy(v,root,read(root/'copy-transformation.json'),True)
            _accounting(v,root,summary);accounting=True
            if valid:
                elapsed=summary['elapsed_seconds'];completion=read(root/'controller-completion.json')
                if (not _number(completion['elapsed_seconds']) or not elapsed<=completion['elapsed_seconds']<v['config']['total_wall_seconds']
                    or completion['status']!='passed' or completion['summary_sha256']!=digest(root/'summary.json')
                    or summary['probe_attempts']!=1 or summary['probe_work_status']!='completed'):raise ValueError('Controller timing/attempt/completion mismatch')
            elif summary['status']=='passed' or summary['verification_valid'] is not False:raise ValueError('Terminal status and validity disagree')
        except (OSError,ValueError,KeyError,TypeError) as exc:valid=False;errors.append(str(exc))
    result={**summary,'role':ROLE,'verification_valid':bool(valid),'original_outcome':original_outcome(v),'original_effort':original_effort(v),
        'work_accounting_valid':accounting,'combined_work_counts_complete':False,'contrasts_available':False,'full_pilot':'pending','verification_errors':errors,
        'source_root':str(v['source_root']),'recovery_root':str(root),'report_generator_sha256':digest(__file__),
        'artifacts_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()},
        'limits':['A passing copy receipt reproduces saved endpoint predictions under the pinned implementation; it is not DFT accuracy or physical recovery.',
                  'Original matrix remains failed, original learning gates failed and three cells blocked; no hybrid matrix, contrasts, winner or pilot clearance.',
                  'Original child work/duration remains unknown; original and recovery effort are separate, never complete combined totals or efficiency evidence.',
                  'No new training, optimizer proposal, warmup, curve remeasurement or held-out evaluation. Future matrix work needs a separate reviewed plan.']}
    if errors and not accounting:
        result['claimed_terminal_record']=summary
        # Quarantine invalid claims. Only independently bound available work may be reported as known.
        result.update(_available_work(v,root),elapsed_seconds=None)
    output.mkdir(parents=True,exist_ok=True);(output/'width-rate-recovery-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Copied-checkpoint recovery verification','',f"Recovery status: **{result.get('status')}**; copied verification valid: **{valid}**.",'',
           f"New training updates: **0**. Numerical verifier attempts: **{result.get('probe_attempts')}**. Completed/unknown new work: `{json.dumps(result.get('new_work'),sort_keys=True)}`.",'',
           f"Controller elapsed: {result.get('elapsed_seconds')} seconds; measured verifier elapsed: {result.get('probe_seconds')} seconds. Later report/archive work is outside the controller cap.",'',
           'The original matrix remains **failed**, both original learning gates **failed**, three cells **blocked**, contrasts **unavailable**, full pilot **pending**. This separate receipt is never grafted onto the old trial.','',
           'Original accounting is inherited completely in the JSON, including per-arm oracle/evaluation/reference work and parent/shared initialization receipts. The old failed verifier duration and unreceipted work remain unavailable; combined accounting is incomplete.','']
    if errors:lines+=['Verification errors:']+['- '+e for e in errors]+['']
    lines+=['- '+limit for limit in result['limits']]
    (output/'width-rate-recovery-verification.md').write_text('\n'.join(lines)+'\n');return result


def generate(root,output):
    config=read('configs/aspirin-width-rate-recovery.json');v=verify_recovery_sources(config)
    if Path(root).resolve()!=Path(config['run_directory']).resolve():raise ValueError('Unreviewed recovery root')
    return _generate(v,root,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('reports'));a=parser.parse_args();r=generate(a.root,a.output);print(json.dumps({k:r[k] for k in ('verification_valid','contrasts_available','full_pilot')},indent=2));raise SystemExit(0 if r['verification_valid'] else 1)
