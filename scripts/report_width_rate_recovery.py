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


def _generate(v,root,output):
    root=Path(root);output=Path(output);errors=[];file=root/'controller-failure.json' if (root/'controller-failure.json').exists() else root/'summary.json';summary=read(file)
    valid=summary.get('verification_valid') is True and summary.get('status')=='passed'
    try:
        if summary['role']!=ROLE or summary['config']!=v['config'] or summary['config_identity']!=canonical_hash(v['config']):raise ValueError('Recovery controls/role changed')
        if summary['original_effort']!=original_effort(v) or summary['original_outcome']!=original_outcome(v):raise ValueError('Original failed outcome/effort changed')
        if summary['code']['files_sha256']!=code_record()['files_sha256']:raise ValueError('Executed numerical/audit code changed')
        _audit_copy(v,root,read(root/'copy-transformation.json'),True)
        if valid:
            elapsed=summary['elapsed_seconds'];completion=read(root/'controller-completion.json')
            if (not isinstance(elapsed,(int,float)) or isinstance(elapsed,bool) or not math.isfinite(elapsed)
                or not 0<=elapsed<=completion['elapsed_seconds']<v['config']['total_wall_seconds']
                or completion['status']!='passed' or completion['summary_sha256']!=digest(root/'summary.json')
                or summary['probe_attempts']!=1 or summary['probe_work_status']!='completed' or summary['full_pilot']!='pending'
                or not math.isfinite(summary['probe_seconds']) or not 0<=summary['probe_seconds']<=elapsed):raise ValueError('Controller timing/attempt/completion mismatch')
            if valid_child(v,root,summary['child_receipt'])!=summary['new_work']:raise ValueError('Completed child work mismatch')
        elif summary.get('probe_attempts') not in (0,1):raise ValueError('Unreviewed additional attempts')
    except (OSError,ValueError,KeyError,TypeError) as exc:valid=False;errors.append(str(exc))
    result={**summary,'role':ROLE,'verification_valid':bool(valid),'original_outcome':original_outcome(v),'original_effort':original_effort(v),
        'combined_work_counts_complete':False,'contrasts_available':False,'full_pilot':'pending','verification_errors':errors,
        'source_root':str(v['source_root']),'recovery_root':str(root),'report_generator_sha256':digest(__file__),
        'artifacts_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
        'limits':['A passing copy receipt reproduces saved endpoint predictions under the pinned implementation; it is not DFT accuracy or physical recovery.',
                  'Original matrix remains failed, original learning gates failed and three cells blocked; no hybrid matrix, contrasts, winner or pilot clearance.',
                  'Original child work/duration remains unknown; original and recovery effort are separate, never complete combined totals or efficiency evidence.',
                  'No new training, optimizer proposal, warmup, curve remeasurement or held-out evaluation. Future matrix work needs a separate reviewed plan.']}
    output.mkdir(parents=True,exist_ok=True);(output/'width-rate-recovery-verification.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Copied-checkpoint recovery verification','',f"Recovery status: **{summary.get('status')}**; copied verification valid: **{valid}**.",'',
           f"New training updates: **0**. Numerical verifier attempts: **{summary.get('probe_attempts')}**. Completed/unknown new work: `{json.dumps(summary.get('new_work'),sort_keys=True)}`.",'',
           f"Controller elapsed: {summary.get('elapsed_seconds')} seconds; measured verifier elapsed: {summary.get('probe_seconds')} seconds. Later report/archive work is outside the controller cap.",'',
           'The original matrix remains **failed**, both original learning gates **failed**, three cells **blocked**, contrasts **unavailable**, full pilot **pending**. This separate receipt is never grafted onto the old trial.','',
           'Original accounting is inherited completely in the JSON, including per-arm oracle/evaluation/reference work and parent/shared initialization receipts. The old failed verifier duration and unreceipted work remain unavailable; combined accounting is incomplete.','']
    if errors:lines+=['Verification errors:']+['- '+e for e in errors]+['']
    lines+=['- '+limit for limit in result['limits']]
    (output/'width-rate-recovery-verification.md').write_text('\n'.join(lines)+'\n');return result


def generate(root,output):
    config=read(Path(root)/'config.json');v=verify_recovery_sources(config)
    if Path(root).resolve()!=Path(config['run_directory']).resolve():raise ValueError('Unreviewed recovery root')
    return _generate(v,root,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('reports'));a=parser.parse_args();r=generate(a.root,a.output);print(json.dumps({k:r[k] for k in ('verification_valid','contrasts_available','full_pilot')},indent=2));raise SystemExit(0 if r['verification_valid'] else 1)
