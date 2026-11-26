"""Saved arrays and serialized states only; no model evaluation or training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from force_consistency import capacity_contract as cc
from train_offset_profile import verify_inputs,verify_gate,projection_binding,probe_binding,validate_launch_code

def leaf(a):
    import msgpack
    shape,dtype,data=msgpack.unpackb(a.data,raw=False)
    return np.frombuffer(data,dtype=dtype).reshape(shape)

def raw_full_hash(raw):
    import msgpack
    from force_consistency.width_rate_replacement import _sorted_tree
    return hashlib.sha256(msgpack.packb(_sorted_tree(raw),use_bin_type=True)).hexdigest()

def decoded(value):
    import msgpack
    if isinstance(value,msgpack.ExtType):return leaf(value)
    if isinstance(value,dict):return {k:decoded(v) for k,v in value.items()}
    return value

def close_arrays(a,b):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Oracle tree structure differs')
        for k in a:close_arrays(a[k],b[k])
    elif np.asarray(a).shape!=np.asarray(b).shape or not np.allclose(a,b,atol=1e-10,rtol=1e-8):raise ValueError('Saved oracle array differs')

def score_arrays(root,p,arrays):
    result={}
    for role,meta in p['inputs'].items():
        with np.load(root/(role+'-inputs.npz'),allow_pickle=False) as targets:
            ids=meta['record_ids'];energy=arrays[role+'_energy'];forces=arrays[role+'_forces'];target=targets['energy_residual'];ft=targets['forces']
            if energy.shape!=target.shape or forces.shape!=ft.shape:raise ValueError('Saved prediction shape differs')
            if not np.isfinite(energy).all() or not np.isfinite(forces).all():raise ValueError('Nonfinite saved predictions')
            def score(indices):
                ii=list(indices);e=energy[ii]-target[ii];f=forces[ii]-ft[ii]
                em=float(np.mean(e**2));fm=float(np.mean(f**2))
                return {'record_count':len(ii),'energy_count':len(ii),'force_component_count':int(ft[ii].size),
                    'record_ids':[ids[i] for i in ii],'energy_mse_eV2':em,'force_mse_eV2_per_A2':fm,
                    'energy_rmse_eV':em**.5,'force_rmse_eV_per_A':fm**.5,
                    'zero_energy_mse_eV2':float(np.mean(target[ii]**2)),'mean_energy_mse_eV2':float(np.var(target[ii])),
                    'zero_force_mse_eV2_per_A2':float(np.mean(ft[ii]**2)),
                    'signed_mean_eV':float(np.mean(e)),'squared_mean_eV2':float(np.mean(e)**2),'centered_population_variance_eV2':float(np.var(e))}
            s=score(range(len(ids)));groups=sorted({g for r in ids for g in meta['memberships'][r]['groups']})
            s['per_group']={g:score(i for i,r in enumerate(ids) if g in meta['memberships'][r]['groups']) for g in groups}
            s['equal_group']={k:float(np.mean([m[k] for m in s['per_group'].values()])) if groups else None for k in cc.MSE_KEYS}
            result[role]=s
    return result

def close_metrics(actual,expected):
    if isinstance(actual,dict):
        if set(actual)!=set(expected):raise ValueError('Saved metric keys differ')
        for k in actual:close_metrics(actual[k],expected[k])
    elif isinstance(actual,float):
        if not np.isclose(actual,expected,atol=1e-12,rtol=1e-10):raise ValueError('Saved metric differs')
    elif actual!=expected:raise ValueError('Saved metric identities differ')

def raw_gate(current,baseline,fraction):
    ratios={k:current[k]/baseline[k] for k in cc.MSE_KEYS}
    return {'accepted':all(np.isfinite(v) and v<fraction for v in ratios.values()),'fraction_remaining':ratios,
        'threshold_mse':{k:baseline[k]*fraction for k in cc.MSE_KEYS},'gate_fraction':fraction}

def audit_saved(root,*,complete=True,source=None,gate=None):
    root=cc.no_symlinks(root);c=cc.read(root/'config.json');p=cc.read(root/'provenance.json')
    if source is None:source=verify_inputs(c)
    if gate is None:gate=verify_gate(c)
    validate_launch_code(cc.read(root/'code.json'),p,gate)
    original=source['provenance']
    if (cc.read(root/'prelaunch-gate.json')!=gate or cc.read(root/'source-pins.json')!=source['pins']
        or p['config_identity']!=cc.canonical_hash(c) or p['gate_identity']!=cc.canonical_hash(gate)
        or p['inputs']!=original['inputs'] or p['normalization']!=original['normalization']
        or p['original_baselines']!=original['original_baselines'] or p['model']!=original['model']
        or p['source_zero_state_hashes']!=original['initial_state_hashes']
        or p['source_blueprint']!=source['zero']['params_schema']
        or cc.read(root/'shared-zero-provenance.json')!=original
        or cc.digest(root/'shared-zero.msgpack')!=cc.digest(Path(source['root'])/'snapshots/0.msgpack')):
        raise ValueError('Saved provenance differs from source/gate')
    for role in p['inputs']:
        if cc.digest(root/(role+'-inputs.npz'))!=p['input_files_sha256'][role] or p['input_files_sha256'][role]!=original['input_files_sha256'][role]:raise ValueError('Input hash differs')
    zero=cc._envelope(root/'shared-zero.msgpack',original)
    import msgpack
    envelope=msgpack.unpackb((root/'shared-zero.msgpack').read_bytes(),raw=False)
    if p['source_zero_full_hash']!=raw_full_hash(zero):raise ValueError('Source all-field hash differs')
    if p['source_zero_payload_sha256']!=envelope['sha256']:raise ValueError('Source serialized payload hash differs')
    receipt=cc.read(root/'commit.json')
    if receipt!=cc.read(root/'commits'/ (str(receipt['update'])+'.json')):raise ValueError('Orphan/unpublished commit')
    if cc.digest(root/receipt['history'])!=receipt['history_sha256']:raise ValueError('History hash differs')
    hist=cc.read(root/receipt['history']);grid=[r['update'] for r in hist]
    if grid!=[u for u in c['score_updates'] if u<=receipt['update']] or (complete and grid!=c['score_updates']):raise ValueError('Score grid differs')
    projected_zero=None;state_bias={};endpoint=None
    for row in hist:
        u=row['update'];commit=cc.read(root/'commits'/ (str(u)+'.json'))
        if commit['row']!=row or commit['update']!=u:raise ValueError('Immutable commit/history row differs')
        for key in ('checkpoint','history','predictions'):
            if commit[key]!={'checkpoint':f'snapshots/{u}.msgpack','history':f'histories/{u}.json','predictions':f'predictions/{u}.npz'}[key] or cc.digest(root/commit[key])!=commit[key+'_sha256']:raise ValueError('Immutable artifact receipt differs')
        if cc.read(root/commit['history'])!=hist[:hist.index(row)+1]:raise ValueError('Immutable history prefix differs')
        raw=cc._envelope(root/commit['checkpoint'],p);meta=cc._state_metadata(raw,u,p['normalization'],source['zero']['params_schema'])
        full=raw_full_hash(raw)
        if full!=row['full_hash'] or full!=commit['full_hash'] or meta['state_hashes']!=row['state_hashes']:raise ValueError('All-field saved state hash differs')
        if any(raw[k]!=zero[k] for k in ('model_rng','sampler_rng_json','normalization_json')):raise ValueError('Source RNG/normalization state changed')
        if any(np.any(leaf(raw['optimizer']['0'][k]['energy_head']['0+']['bias'])!=0) for k in ('mu','nu')):raise ValueError('Saved bias moments nonzero')
        state_bias[u]=float(leaf(raw['params']['energy_head']['0+']['bias'])[0])
        if u==0:projected_zero=raw
        with np.load(root/commit['predictions'],allow_pickle=False) as data:recomputed=score_arrays(root,p,data)
        close_metrics(recomputed,row['metrics']);endpoint=recomputed
        endpoint_gates={r:raw_gate(recomputed[r],original['original_baselines'][r],c['gate_fraction']) for r in ('fitting','anchor')}
        for role in endpoint_gates:
            if endpoint_gates[role]!=row[role+'_gate']:raise ValueError('Saved raw gate decision differs')
        if cc.digest(root/row['projection'])!=row['projection_sha256']:raise ValueError('Score projection link differs')
        if u:
            ap=root/row['accepted_receipt']
            if cc.digest(ap)!=row['accepted_receipt_sha256'] or cc.read(ap)['outgoing_full_hash']!=full:raise ValueError('Score accepted-state receipt differs')
    # Initial projection can change only the scalar bias; all optimizer and RNG fields remain source-identical.
    import copy
    adjusted=copy.deepcopy(projected_zero);adjusted['params']['energy_head']['0+']['bias']=zero['params']['energy_head']['0+']['bias']
    if adjusted!=zero or hist[0]['state_hashes']!=p['initial_state_hashes'] or hist[0]['full_hash']!=p['initial_full_hash']:raise ValueError('Initial projection changed nonbias/source state')
    replay=cc.read(root/'shared-zero-replay.json')
    with np.load(root/'replay/shared-zero.npz',allow_pickle=False) as data:close_metrics(score_arrays(root,p,data),replay)
    for role in ('fitting','anchor'):
        for k in cc.MSE_KEYS:
            if not np.isclose(replay[role][k],original['original_baselines'][role][k],atol=1e-12,rtol=1e-10):raise ValueError('Original baseline replay differs')
    corrections=[];previous=p['initial_full_hash'];previous_bias=float(leaf(zero['params']['energy_head']['0+']['bias'])[0])
    for u in range(receipt['update']+1):
        phase='initial' if u==0 else 'training';name=f'projections/{phase}-{u}'
        record=cc.read(root/(name+'.json'));expected=projection_binding(p,gate,phase,max(0,u-1),u)
        if any(record.get(k)!=v for k,v in expected.items()) or record['vector_sha256']!=cc.digest(root/(name+'.npz')):raise ValueError('Projection vector/identity binding differs')
        with np.load(root/(name+'.npz'),allow_pickle=False) as data:
            e=data['errors'];atoms=data['atom_counts']
            if len(e)!=len(p['inputs']['fitting']['record_ids']) or not np.isfinite(e).all() or np.any(atoms!=c['atom_count']):raise ValueError('Projection support/nonfinite differs')
            delta=-float(np.mean(e))/c['atom_count']
            calculated={'delta_bias':delta,'preprojection_mean_eV':float(np.mean(e)),
                'preprojection_mse_eV2':float(np.mean(e**2)),'centered_variance_eV2':float(np.var(e)),
                'predicted_post_mean_eV':float(np.mean(e+atoms*delta)),
                'predicted_post_mse_eV2':float(np.mean((e+atoms*delta)**2))}
            for k,v in calculated.items():
                if not np.isclose(v,record[k],atol=1e-12,rtol=1e-10):raise ValueError('Projection statistic differs')
        if record['bias_before']!=previous_bias or not np.isclose(record['bias_after'],previous_bias+delta,atol=1e-14,rtol=0):raise ValueError('Projection bias chain differs')
        previous_bias=record['bias_after']
        if u in state_bias and state_bias[u]!=previous_bias:raise ValueError('Saved state bias differs from projection')
        if u==0:
            if record['incoming_full_hash']!=p['source_zero_full_hash']:raise ValueError('Initial projection incoming state differs')
        else:
            accepted=cc.read(root/'accepted'/ (str(u)+'.json'))
            if (accepted['projection']!=name+'.json' or accepted['projection_sha256']!=cc.digest(root/(name+'.json'))
                or accepted['incoming_full_hash']!=previous or record['incoming_full_hash']!=previous
                or accepted['incoming_update']!=u-1 or accepted['outgoing_update']!=u
                or accepted['ordinal']!=1 or accepted['config_identity']!=p['config_identity'] or accepted['gate_identity']!=p['gate_identity']
                or accepted['projection'] not in (name+'.json',)):
                raise ValueError('Accepted update/projection chain differs')
            previous=accepted['outgoing_full_hash']
        corrections.append(record)
    if previous!=receipt['full_hash']:raise ValueError('Endpoint accepted chain differs')
    for u in c['oracle_updates']:
        if u>receipt['update']:continue
        oracle=cc.read(root/'oracles'/ (str(u)+'.json'))
        expected=projection_binding(p,gate,'oracle-reference',u-1,u)
        if oracle['projection_binding']!=expected or oracle['status']!='passed':raise ValueError('Oracle identity differs')
        actual_incoming=p['initial_full_hash'] if u==1 else cc.read(root/'accepted'/ (str(u-1)+'.json'))['outgoing_full_hash']
        if oracle['incoming_update']!=u-1 or oracle['outgoing_update']!=u or oracle['incoming_full_hash']!=actual_incoming:raise ValueError('Oracle incoming state differs')
        members={str(u)+'-'+n for n in ('oracle_gradient.msgpack','oracle_numpy_adam.msgpack','oracle_prediction_reference.msgpack','oracle_prediction_actual.msgpack','oracle_complete.msgpack','oracle_complete.json','oracle_reference_projection.msgpack','oracle_reference_projection.json','projection.npz','projection.json')}
        if set(oracle['artifact_sha256'])!=members:raise ValueError('Incomplete oracle artifacts')
        for n,sha in oracle['artifact_sha256'].items():
            if cc.digest(root/'oracles'/n)!=sha:raise ValueError('Oracle artifact hash differs')
        raw_data=msgpack.unpackb((root/'oracles'/ (str(u)+'-oracle_complete.msgpack')).read_bytes(),raw=False)
        data=decoded(raw_data)
        close_arrays(data['reference_gradient'],data['actual_gradient'])
        raw=decoded(cc._envelope(root/'snapshots'/ (str(u)+'.msgpack'),p))
        close_arrays(data['reference_params'],raw['params']);close_arrays(data['reference_optimizer'],raw['optimizer'])
        close_arrays(data['reference_energy_adjusted'],data['actual_energy']);close_arrays(data['reference_forces'],data['actual_forces'])
        if oracle['candidate_full_hash']!=cc.read(root/'accepted'/ (str(u)+'.json'))['outgoing_full_hash']:raise ValueError('Oracle accepted state differs')
        projection=cc.read(root/'oracles'/ (str(u)+'-projection.json'))
        if any(projection[k]!=v for k,v in expected.items()) or projection['incoming_full_hash']!=actual_incoming or projection['actual_candidate_full_hash']!=oracle['candidate_full_hash']:raise ValueError('Oracle projection state binding differs')
        if projection['vector_sha256']!=cc.digest(root/'oracles'/ (str(u)+'-projection.npz')):raise ValueError('Oracle projection NPZ hash differs')
        raw_proposal=msgpack.unpackb((root/'oracles'/ (str(u)+'-oracle_numpy_adam.msgpack')).read_bytes(),raw=False)
        reference_raw={**zero,'params':raw_proposal['0'],'optimizer':raw_proposal['1'],'update':u}
        reference_projected={**zero,'params':raw_data['reference_params'],'optimizer':raw_data['reference_optimizer'],'update':u}
        if raw_full_hash(reference_raw)!=projection['reference_raw_full_hash'] or raw_full_hash(reference_projected)!=projection['reference_projected_full_hash']:raise ValueError('Oracle reference all-field identity differs')
        if projection['bias_before']!=float(leaf(reference_raw['params']['energy_head']['0+']['bias'])[0]) or projection['bias_after']!=float(leaf(reference_projected['params']['energy_head']['0+']['bias'])[0]):raise ValueError('Oracle reference bias differs')
        with np.load(root/'oracles'/ (str(u)+'-projection.npz'),allow_pickle=False) as vector:
            with np.load(root/'fitting-inputs.npz',allow_pickle=False) as target:errors=data['reference_energy_raw']-target['energy_residual']
            if not np.array_equal(errors,vector['errors']) or np.any(vector['atom_counts']!=c['atom_count']):raise ValueError('Oracle projection vector differs')
            delta=-float(np.mean(errors))/c['atom_count'];shift=errors+c['atom_count']*delta
            for k,v in {'delta_bias':delta,'preprojection_mean_eV':float(np.mean(errors)),'preprojection_mse_eV2':float(np.mean(errors**2)),
                'centered_variance_eV2':float(np.var(errors)),'predicted_post_mean_eV':float(np.mean(shift)),
                'predicted_post_mse_eV2':float(np.mean(shift**2))}.items():
                if not np.isclose(projection[k],v,atol=1e-12,rtol=1e-10):raise ValueError('Oracle projection statistics differ')
    if complete:
        s=cc.read(root/'summary.json');child=cc.read(root/'checkpoint-verification.json');completion=cc.read(root/'probe-completion.json');entry=cc.read(root/'probe-entry.json');attempt=cc.read(root/'probe-attempt.json')
        if (not s['execution_valid'] or s['status']!='completed' or s['updates']!=c['max_updates'] or s['accepted_calibrations']!=c['max_updates']+1
            or s['pending_inflight_unknown'] is not None or s['pending_update_not_accepted'] or s['late_call']
            or s['elapsed_seconds']>=c['total_wall_seconds'] or s['endpoint']!=hist[-1]
            or s['checkpoint_verification']!=child or child['status']!='passed' or child['binding']!=probe_binding(root)
            or child['maximum_absolute_error']>1e-10 or child['prediction_frames']!=144 or child['random_initializations']!=0 or child['training_updates']!=0
            or child['full_hash']!=receipt['full_hash'] or child['state_hashes']!=receipt['state_hashes'] or child['adam_count']!=6000
            or completion['status']!='passed' or completion['verification_sha256']!=cc.digest(root/'checkpoint-verification.json')
            or entry['attempt_sha256']!=cc.digest(root/'probe-attempt.json') or completion['attempt_sha256']!=entry['attempt_sha256'] or attempt['binding']!=child['binding']
            or cc.read(root/'probe-process.json')['exit_code']!=0):raise ValueError('Completed run/child contract differs')
        for u in c['oracle_updates']:
            oracle=cc.read(root/'oracles'/ (str(u)+'.json'))
            if oracle!=s['oracles'][str(u)] or oracle['status']!='passed' or (u>1 and not oracle['incoming_nonzero_moments']):raise ValueError('Production oracle missing/invalid')
        work=s['work']
        if work['training:gradient']['valid_frames']!=c['max_updates']*128 or work['training:adam_proposal']['valid_calls']!=c['max_updates'] or work['training:calibration_energy']['valid_frames']!=c['max_updates']*128:raise ValueError('Accepted work counts differ')
    return {'scored_states':len(hist),'accepted_projection_receipts':len(corrections),'recomputed_endpoint':endpoint,
        'history':hist,'recomputed_gates':endpoint_gates,'corrections':{'scope':'dense intervention only; historical reference is sparse',
            'maximum_absolute_delta_bias':max(abs(r['delta_bias']) for r in corrections),
            'maximum_absolute_preprojection_mean_eV':max(abs(r['preprojection_mean_eV']) for r in corrections),
            'initial_delta_bias':corrections[0]['delta_bias']},'source_zero_full_hash':p['source_zero_full_hash']}

def matching_rows(reference,intervention):
    a={r['update']:r for r in reference if r['update']>0};b={r['update']:r for r in intervention if r['update']>0}
    return [(u,a[u],b[u]) for u in sorted(a.keys()&b.keys())]

def incomplete_report(root,reason='Missing completion summary'):
    path=Path(root)/'summary.json';summary=cc.read(path) if path.exists() else None
    return {'execution_valid':False,'scientific_endorsement':False,'status':'incomplete_or_invalid',
        'reason':reason,'retained_summary':summary,'next_action':'resolve_verification_or_operational_failure; no research retry'}

def build_report(root):
    root=Path(root)
    if not (root/'summary.json').exists():return incomplete_report(root)
    try:
        audit=audit_saved(root);c=cc.read(root/'config.json');p=cc.read(root/'provenance.json');s=cc.read(root/'summary.json')
        reference_root=Path(c['source_root']);commit=cc.read(reference_root/'commit.json');reference=cc.read(reference_root/commit['history'])
        pairs=matching_rows(reference,audit['history'])
        if len(pairs)!=62:raise ValueError('Expected62 shared postzero observations')
        contrasts={}
        for role in ('fitting','anchor'):
            contrasts[role]={}
            for key in cc.MSE_KEYS:
                base=p['original_baselines'][role][key];ref=reference[-1]['metrics'][role][key];new=audit['history'][-1]['metrics'][role][key]
                contrasts[role][key]={'original_baseline':base,'historical_endpoint':ref,'intervention_endpoint':new,
                    'historical_fraction':ref/base,'intervention_fraction':new/base,'relative_endpoint_change':new/ref-1,
                    'historical_matching_grid_maximum':max(a['metrics'][role][key] for _,a,b in pairs),
                    'intervention_matching_grid_maximum':max(b['metrics'][role][key] for _,a,b in pairs)}
        gates=audit['recomputed_gates']
        return {'execution_valid':True,'scientific_endorsement':True,'status':'verified_completed',
            'learning_gate_passed':all(g['accepted'] for g in gates.values()),'gates':gates,'contrasts':contrasts,
            'matching_postzero_observations':len(pairs),'audit':audit,'summary':s,
            'limitations':['Mean-zero fitting bias is enforced by construction.','One seed/rate, historical reference and changed optimizer algorithm limit causal claims.',
                'All128 fitting examples and16 fitted anchors provide no held-out generalization evidence.',
                'Sparse score observations do not prove continuous stability or convergence.','Dense intervention corrections have no matching dense historical reference.',
                'Full pilot remains pending.']}
    except Exception as exc:return incomplete_report(root,type(exc).__name__+': '+str(exc))

def render(root,outdir):
    report=build_report(root);outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'offset-profile-diagnostics.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    lines=['# Controlled energy-offset profiling','',f"Execution verified: {report['execution_valid']}. Learning gate: {report.get('learning_gate_passed','unendorsed')}.",'']
    if report['execution_valid']:
        lines+=['Historical width32 comparison; original unprojected zero-state MSE baselines.','',
            '| Role | Metric | Historical endpoint fraction | Profiled endpoint fraction | Relative MSE change |','|---|---|---:|---:|---:|']
        for role,metrics in report['contrasts'].items():
            for key,v in metrics.items():lines.append(f"| {role} | {key} | {v['historical_fraction']:.12g} | {v['intervention_fraction']:.12g} | {v['relative_endpoint_change']:.12g} |")
        lines+=['','62 shared postzero observations; dense correction statistics reported separately.','']+report['limitations']
    else:lines+=[report['reason'],'Observations retained; numerical comparison is not endorsed.']
    (outdir/'offset-profile-diagnostics.md').write_text('\n'.join(lines)+'\n')
    return report

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',default='runs/aspirin-offset-profile-2026-10-02');a.add_argument('--outdir',default='reports');n=a.parse_args()
    r=render(n.root,n.outdir);return 0 if r['execution_valid'] else 1
if __name__=='__main__':raise SystemExit(main())
