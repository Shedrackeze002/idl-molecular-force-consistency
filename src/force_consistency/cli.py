import argparse
from dataclasses import asdict
from pathlib import Path
import json

from .data.types import PreparationPolicy
from .data.xyz import read_records
from .data.audit import inventory_connections
from .data.matching import audit_duplicates,map_endpoints
from .data.prepare import export_prepared,write_json
from .data.units import verify_units,convert_record


def main(argv=None):
    parser=argparse.ArgumentParser(description='Audited Landscape17 preparation and baseline experiments')
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('train',help='Run a capped, verified energy baseline')
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--backend',choices=('cpu','gpu'),help='Explicit device requirement; unavailable GPU fails')
    p=subs.add_parser('diagnose-width-rate',help='Run the frozen fresh model-width and fixed-rate matrix')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-convergence-budget',help='Continue three verified saved states with one fixed training budget')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-optimizer-step',help='Inspect frozen saved Adam states and fixed local probes')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-loss-balance',help='Run reviewed fixed loss-weight controls')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-gradients',help='Inspect frozen representation and loss gradients')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-coverage',help='Run reviewed training-only coverage matrix')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('diagnose-learning',help='Run reviewed training-only learning diagnostics')
    p.add_argument('--config',required=True,type=Path)
    p=subs.add_parser('summarize-run',help='Read saved run results without accessing test data')
    p.add_argument('--run',required=True,type=Path)
    for name in ('audit','prepare'):
        p=subs.add_parser(name)
        p.add_argument('--archive',type=Path,default=Path('data/Landscape17.zip'))
        p.add_argument('--policy',type=Path,default=Path('configs/preparation.json'))
        p.add_argument('--unit-evidence',type=Path,default=Path('manifests/landscape17-unit-evidence.json'))
        p.add_argument('--output',type=Path)
        if name=='prepare': p.add_argument('--molecule',required=True,choices=('aspirin','paracetamol'))
    args=parser.parse_args(argv)
    if args.command=='diagnose-width-rate':
        from .width_rate_runs import diagnose_width_rate
        summary=diagnose_width_rate(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-convergence-budget':
        from .convergence_runs import diagnose_convergence
        summary=diagnose_convergence(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','learning_accepted','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-optimizer-step':
        from .optimizer_runs import diagnose_optimizer_steps
        summary=diagnose_optimizer_steps(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','next_step','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-loss-balance':
        from .loss_balance_runs import diagnose_loss_balance
        summary=diagnose_loss_balance(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','balanced_joint_accepted','next_step','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-gradients':
        from .gradient_runs import diagnose_gradients
        summary=diagnose_gradients(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','next_step','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-coverage':
        from .coverage_matrix import diagnose_coverage
        summary=diagnose_coverage(json.loads(args.config.read_text()))
        print(json.dumps({k:summary.get(k) for k in ('status','execution_valid','next_step','full_pilot','failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='diagnose-learning':
        from .diagnostics import diagnose
        summary=diagnose(json.loads(args.config.read_text()))
        print(json.dumps({'status':summary['status'],'teacher':summary['teacher'].get('status'),'real':summary['real'].get('status'),'full_pilot':summary['full_pilot'],'failure_reason':summary.get('failure_reason')},indent=2))
        return 0 if summary['execution_valid'] else 1
    if args.command=='train':
        from .train import train
        config=json.loads(args.config.read_text())
        if args.backend: config['backend']=args.backend
        summary=train(config)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('provenance','initial_validation','final_validation')},indent=2))
        return 0 if summary['status']=='passed' else 1
    if args.command=='summarize-run':
        print(json.dumps(json.loads((args.run/'summary.json').read_text()),indent=2))
        return 0
    policy=PreparationPolicy(**json.loads(args.policy.read_text()))
    evidence=json.loads(args.unit_evidence.read_text());units=verify_units(evidence)
    molecules=(args.molecule,) if args.command=='prepare' else ('aspirin','paracetamol')
    rs=[convert_record(r,units) for r in read_records(args.archive,molecules)]
    inv=inventory_connections(rs);ends=map_endpoints(inv,rs,policy);dups=audit_duplicates(rs,policy)
    if args.command=='prepare':
        out=args.output or Path('data/processed')/args.molecule
        manifest=export_prepared(rs,directory=out,inventory=inv,endpoints=ends,duplicates=dups,policy=policy,molecule=args.molecule,unit_evidence=evidence)
        print(json.dumps({'status':'prepared','molecule':args.molecule,'identity':manifest['identity'],'split_hash':manifest['split_hash'],'counts':{k:len(v['record_ids']) for k,v in manifest['partitions'].items()}},indent=2))
    else:
        out=args.output or Path('reports/audit');out.mkdir(parents=True,exist_ok=True)
        write_json(out/'endpoints.json',ends);write_json(out/'duplicates.json',asdict(dups));write_json(out/'inventory.json',asdict(inv))
        print(json.dumps({'counts':inv.counts,'endpoint_statuses':{m:{s:sum(v['status']==s for k,v in ends.items() if k.startswith(m+'/')) for s in ('matched','unmatched','ambiguous','unresolved')} for m in molecules}},indent=2))


if __name__=='__main__': raise SystemExit(main())
