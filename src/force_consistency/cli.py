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
    for name in ('audit','prepare'):
        p=subs.add_parser(name)
        p.add_argument('--archive',type=Path,default=Path('data/Landscape17.zip'))
        p.add_argument('--policy',type=Path,default=Path('configs/preparation.json'))
        p.add_argument('--unit-evidence',type=Path,default=Path('manifests/landscape17-unit-evidence.json'))
        p.add_argument('--output',type=Path)
        if name=='prepare': p.add_argument('--molecule',required=True,choices=('aspirin','paracetamol'))
    args=parser.parse_args(argv)
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


if __name__=='__main__': main()
