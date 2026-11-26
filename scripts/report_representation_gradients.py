"""Generate a descriptive report from preserved observations, with no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def generate(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=read(root/'summary.json');phase=read(root/'phase-A.json');config=read(root/'config.json')
    names=['fresh-A-fitting','fresh-B-fitting','fresh-C-fitting','final-A-fitting','final-B-fitting','final-C-fitting','final-B-anchor','final-C-anchor']
    observed={name:read(root/name/'observations.json') for name in names if (root/name/'observations.json').exists()}
    attempted=sum(bool(case.get('executed')) for case in summary['cases'].values())
    verified=bool(summary['execution_valid'] and phase['status']=='completed' and
                  all(n in observed and observed[n]['verification_valid'] and summary['cases'][n]['status']=='completed' for n in names))
    compact={'execution':{k:v for k,v in summary.items() if k!='cases'},'constraints':{},'cases':{},
             'sources':{str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
             'generator_sha256':digest(__file__),'matplotlib_version':matplotlib.__version__,'figures':{}}
    lines=['# Frozen representation and loss-gradient diagnostics','',
           f"Execution status: **{summary['status']}**; {summary['elapsed_seconds']:.3f}s. Learning acceptance remains false; full pilot pending.",'',
           f"The protocol fixes the16/16/128 training subsets, source2000-update checkpoints, model and original normalized loss. {attempted} of 8 prespecified cases were attempted in order under600s; it permits no optimizer update, training, test/validation access or label substitution.", '',
           f"Phase A status: **{phase['status']}**.",'',
           '## Necessary force constraints','',
           '| Set | Zero-force MSE | Translation floor | Rigid floor | Original force gate | Obstructs gate |','|---|---:|---:|---:|---:|---|']
    for arm in ('narrow16','diverse16','diverse128'):
        if arm not in phase.get('subsets',{}):continue
        d=phase['subsets'][arm]
        g=d['pooled'];graphs=[r['graph'] for r in d['records']]
        compact['constraints'][arm]={'pooled':g,'equal_group':d['equal_group'],'record_count':d['record_count'],
            'force_component_count':d['force_component_count'],'immutable_force_gate_mse':d['immutable_force_gate_mse'],
            'floor_obstructs_gate':d['floor_obstructs_gate'],'ranks':sorted(set(r['rank'] for r in d['records'])),
            'graph':{'component_range':[min(x['connected_components'] for x in graphs),max(x['connected_components'] for x in graphs)],
                     'isolated_atom_range':[min(x['isolated_atoms'] for x in graphs),max(x['isolated_atoms'] for x in graphs)],
                     'two_hop_fraction_range':[min(x['two_hop_fraction'] for x in graphs),max(x['two_hop_fraction'] for x in graphs)],
                     'cutoff_near_pair_total':sum(x['pairs_near_cutoff'] for x in graphs)}}
        lines.append(f"| {arm} | {g['raw_mse']:.12g} | {g['translation_mse']:.12g} | {g['floor_mse']:.12g} | {d['immutable_force_gate_mse']:.12g} | {d['floor_obstructs_gate']} |")
    lines+=['','MSE units are eV²/Å², pooled over force components. Equal-group values, per-frame net forces/centered torques, ranks and SVD/graph checks are retained separately in phase-A.json. The rigid floor is a numerical necessary-condition bound. A floor below the gate rules out these constraints alone as the explanation; it does not prove representability, integrability or sufficient capacity. A floor above it identifies incompatibility under the enforced symmetry, not corrupted labels or a particular calculator error. The raw labels and learning gates remain authoritative.','',
            'Cutoff connectivity and two-hop reach are descriptive support measures; they do not establish expressive sufficiency. The model sums scalar atom energies and obtains forces from negative coordinate derivatives, following the [official E3x force example](https://e3x.readthedocs.io/stable/examples/md17_ethanol.html); its benchmark accuracy/configuration is not project evidence.','',
            '## Frozen gradients','',
            '| Case | Status | Scaled E loss | Scaled F loss | E gradient norm | F gradient norm | F/E norm | Cosine | E slope | F slope |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.8g}'
    for name in names:
        s=summary['cases'][name];d=observed.get(name)
        if d:
            g=d['geometry']['whole'];compact['cases'][name]={'status':s['status'],'record_count':d['record_count'],
                'force_component_count':d['force_component_count'],'update':d['update'],'losses':d['losses'],'geometry':d['geometry'],
                'joint_gradient_identity':d['joint_gradient_identity'],'finite_differences':d['finite_differences'],
                'state_hashes_before':d['state_hashes_before'],'state_hashes_after':d['state_hashes_after'],'verification_valid':d['verification_valid']}
            nums=[d['losses']['energy'],d['losses']['force'],g['energy_norm'],g['force_norm'],g['force_over_energy_norm'],g['cosine'],g['energy_raw_joint_descent_slope'],g['force_raw_joint_descent_slope']]
            lines.append(f"| {name} | {s['status']} | "+' | '.join(fmt(v) for v in nums)+' |')
        else:
            compact['cases'][name]=s;lines.append(f"| {name} | {s['status']} | — | — | — | — | — | — | — | — |")
    lines+=['','Norms, ratios and cosines depend on this parameterization and the fixed training scales. Slopes are local directional derivatives along ordinary **minus the joint gradient**, without normalizing that direction: −(||gE||²+gE·gF), −(||gF||²+gE·gF). Negative slopes imply a local first-order decrease for that objective. They are not the actual Adam direction, a training trajectory or a causal explanation. Zero-norm blocks have undefined cosine; the force head-bias gradient and initial zero-head backbone force gradient are structurally zero. Nonzero later derivatives do not establish adequate capacity.','',
            'Both central-difference steps1e-4 and1e-5 are retained for every completed objective; at least one must agree within1e-8+1e-4 relative tolerance. These checks verify selected directions, not every gradient component. The separate gradient sum is checked against the unchanged joint objective within1e-10/1e-8. Parameter and optimizer hashes before/after are retained.','',
            ]
    objectives=[d['finite_differences'][objective] for d in observed.values() for objective in ('energy','force')]
    passed=sum(bool(o['valid'] and any(step['valid'] for step in o['steps'])) for o in objectives)
    failed=[f"{name}/{objective}/h={step['step']}: error {step['absolute_error']:.12g}" for name,d in observed.items() for objective in ('energy','force') for step in d['finite_differences'][objective]['steps'] if not step['valid']]
    lines += [f"{passed} of {len(objectives)} recorded objectives meet the prespecified at-least-one-step criterion. Failed individual finite-difference steps: "+('; '.join(failed) if failed else 'none recorded')+'. All failed steps remain preserved.','']
    if not verified:
        lines += ['Unverified observations are retained in the tables and raw artifacts; verified scientific interpretation is withheld until verification/operational failures are resolved.','']
    if verified:
        lines += ['## Measured observations','']
        fits=[observed[n]['geometry']['whole'] for n in names[3:6]]
        ratios=[observed[n]['geometry']['whole']['force_over_energy_norm'] for n in names[:3]]
        inverse=[fmt(1/v) if v else 'undefined' for v in ratios]
        lines += [f"Initial energy/force gradient norm ratios for A/B/C: {', '.join(inverse)}. At the final fitting snapshots the cosines are {fits[0]['cosine']:.12g}, {fits[1]['cosine']:.12g}, {fits[2]['cosine']:.12g}. Raw joint-descent energy slopes for A/B/C: {', '.join(fmt(g['energy_raw_joint_descent_slope']) for g in fits)}; force slopes: {', '.join(fmt(g['force_raw_joint_descent_slope']) for g in fits)}. Positive slopes denote a local first-order increase for that objective. Final C anchor cosine: {fmt(observed['final-C-anchor']['geometry']['whole']['cosine'])}. No Adam direction or causal explanation was measured.",'']
    if phase['status']=='completed' and compact['constraints']:
        ranks=sorted({rank for d in compact['constraints'].values() for rank in d['ranks']})
        graphs=[r['graph'] for d in phase['subsets'].values() for r in d['records']]
        lines += [f"Observed rigid ranks: {ranks}. Graph connected-component range: {min(g['connected_components'] for g in graphs)}–{max(g['connected_components'] for g in graphs)}; isolated-atom range: {min(g['isolated_atoms'] for g in graphs)}–{max(g['isolated_atoms'] for g in graphs)}; two-hop reach fraction range: {min(g['two_hop_fraction'] for g in graphs):.12g}–{max(g['two_hop_fraction'] for g in graphs):.12g}. These observations do not establish that this width/degree/message representation can express the target force field.",'']
    lines += ['## Outcome and limits','',
            f"Next decision: `{summary.get('next_step')}`. Static observations do not clear the previous failed force gates. Any architecture, weighting, optimization, data-convention intervention or larger training run requires its own saved plan and two independent reviews. No additional attempt was selected after this result. Unresolved, failed or blocked observations remain in the raw artifacts and paper evidence log.",'',
            '## Reproduction and evidence','',
            f"Analysis artifacts: `{root}`; historical source run: `{config["source_run"]}`. Frozen config: `configs/aspirin-representation-gradient-diagnostic.json`; source/prepared/split/subset identities and historical/current provenance are in the run artifacts. This report generator reads JSON only; it does not load data or execute a model. Raw observations and every partial/failure artifact are indexed by SHA256 in the companion JSON. See `reports/PAPER_EVIDENCE_LOG.md`, entries PLAN-GRADIENT/IMPL-GRADIENT/GRADIENT-*. Results are training-only diagnostics, with no held-out scientific comparison."]
    if phase.get('subsets'):
        arms=[a for a in ('narrow16','diverse16','diverse128') if a in phase['subsets']];fig,ax=plt.subplots(figsize=(7.5,4.5));x=np.arange(len(arms));width=.24
        for offset,key,label in [(-width,'raw_mse','Zero-force baseline'),(0,'floor_mse','Necessary rigid floor')]:
            ax.bar(x+offset,[phase['subsets'][a]['pooled'][key] for a in arms],width,label=label)
        ax.bar(x+width,[phase['subsets'][a]['immutable_force_gate_mse'] for a in arms],width,label='Original force gate')
        ax.set_yscale('log');ax.set_xticks(x,arms);ax.set_ylabel('Force MSE (eV²/Å²)');ax.legend();fig.tight_layout()
        f=out/'representation-force-floors.png';fig.savefig(f,dpi=180);plt.close(fig);compact['figures'][f.name]=digest(f)
    if observed:
        ordered=[n for n in names if n in observed];x=np.arange(len(ordered));fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for key,label in [('energy_norm','Energy'),('force_norm','Force')]:axes[0].scatter(x,[observed[n]['geometry']['whole'][key] for n in ordered],label=label)
        axes[0].set_yscale('log');axes[0].set_ylabel('Scaled-objective gradient L2 norm');axes[0].legend()
        axes[1].scatter(x,[observed[n]['geometry']['whole']['cosine'] if observed[n]['geometry']['whole']['cosine'] is not None else np.nan for n in ordered]);axes[1].axhline(0,color='gray',lw=1)
        axes[1].set_ylim(-1.05,1.05);axes[1].set_ylabel('Energy/force gradient cosine');axes[1].set_xticks(x,ordered,rotation=28,ha='right');fig.tight_layout()
        f=out/'representation-gradient-geometry.png';fig.savefig(f,dpi=180);plt.close(fig);compact['figures'][f.name]=digest(f)
    report=out/'representation-gradient-diagnostics.md';report.write_text('\n'.join(lines)+'\n');compact['report_sha256']=digest(report)
    (out/'representation-gradient-diagnostics.json').write_text(json.dumps(compact,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'report':str(report),'execution_valid':summary['execution_valid'],'figures':compact['figures']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/aspirin-representation-gradient-2026-09-30');p.add_argument('--output',default='reports');a=p.parse_args();generate(a.run,a.output)
