"""Independent arithmetic/file audit. Never calls a model or differentiates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import traceback
import zipfile
import numpy as np
from flax import serialization

BASE = Path.cwd()
OUT = BASE / 'reports/execution/spike-diagnostic-2026-10-02/saved-result-review'
ROOT = BASE / 'runs/aspirin-capacity-spike-2026-10-02'
SOURCE = BASE / 'runs/aspirin-capacity-2026-10-02'
RECORDS = OUT.parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def close(x, y):
    assert np.allclose(x, y, atol=1e-10, rtol=1e-8), (x, y)
def tree(v):
    if isinstance(v, dict): return {k: tree(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)): return type(v)(tree(x) for x in v)
    return v
def leaves(v, prefix=''):
    if isinstance(v, dict):
        return {p: a for k in sorted(v) for p, a in leaves(v[k], prefix + '[' + repr(k) + ']').items()}
    return {prefix: np.asarray(v)}
def pool(e, f):
    mean = float(e.mean()); mse = float(np.mean(e**2)); var = float(np.var(e))
    return {'signed_mean_energy_error_eV': mean, 'mean_energy_error_squared_eV2': mean**2,
            'centered_energy_error_variance_eV2': var, 'energy_mse_eV2': mse,
            'force_mse_eV2_per_A2': float(np.mean(f**2)), 'offset_share': mean**2/mse if mse else None}
def geometry(e, f, a, b, weighted):
    el = leaves(e); fl = leaves(f); assert el.keys() == fl.keys()
    en2 = sum(float(np.vdot(el[k], el[k])) for k in el)
    fn2 = sum(float(np.vdot(fl[k], fl[k])) for k in fl)
    dot = sum(float(np.vdot(el[k], fl[k])) for k in el)
    en = en2**.5; fn = fn2**.5
    d = {'energy_norm': en, 'force_norm': fn, 'dot': dot,
         'parameter_count': sum(v.size for v in el.values()), 'cosine': dot/(en*fn) if en and fn else None,
         'force_over_energy_norm': fn/en if en else None}
    prefix = 'weighted_component_negative_weighted_joint' if weighted else 'scaled_loss_negative_unweighted_joint'
    d['energy_' + prefix + '_slope'] = -(en2 + dot)
    d['force_' + prefix + '_slope'] = -(fn2 + dot)
    if not weighted:
        d['energy_scaled_loss_negative_weighted_joint_slope'] = -(a*en2 + b*dot)
        d['force_scaled_loss_negative_weighted_joint_slope'] = -(a*dot + b*fn2)
    return d
def check_fields(actual, expected):
    for k,v in expected.items():
        if v is None: assert actual[k] is None
        else: close(actual[k], v)
def inventory(root, mapping):
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(mapping)
    for n,s in mapping.items(): assert not (root/n).is_symlink() and sha(root/n)==s
def main():
    result = {'status': 'running', 'new_model_calls': 0, 'new_training_updates': 0,
              'new_initializations': 0, 'new_optimizer_proposals': 0, 'cases': {}, 'failed_fd_steps': []}
    config=read(ROOT/'config.json'); report=read(BASE/'reports/capacity-spike-diagnostics.json')
    source_map=read(BASE/config['source_inventory']); raw_map=read(RECORDS/'numerical-run-inventory.json')
    inventory(SOURCE,source_map); inventory(ROOT,raw_map)
    assert len(source_map)==343 and len(raw_map)==247
    assert sha(BASE/config['source_inventory'])==config['source_inventory_sha256']
    pins=read(BASE/config['source_pins']);assert len(pins)==347
    assert set(pins)=={config['source_root']+'/'+n for n in source_map}|{config['capacity_config'], 'reports/capacity-diagnostics.json','reports/capacity-diagnostics.md','reports/execution/capacity-2026-10-02/source-pins.json'}
    for n,s in pins.items(): assert sha(BASE/n)==s
    gate=read(ROOT/'prelaunch-gate.json');assert gate==read(BASE/config['prelaunch_gate']) and gate['status']=='verified'
    for n,s in gate['artifacts'].items():assert sha(BASE/n)==s
    code=read(ROOT/'analysis-code.json');assert code['revision']=='b013acdae3f4566a8b41d7b417bf43ebe7583c5f'
    assert code['historical_revision']=='b45ff2a66923af14a51dce7190b3808810b2a478'
    for n,s in code['code_files'].items():assert sha(BASE/n)==s
    assert sha(BASE/'scripts/diagnose_capacity_spikes.py')==code['script_sha256']
    from force_consistency import capacity_contract as cc
    cc.verify_code(code);cc.verify_sources(read(BASE/config['capacity_config']))
    summary=read(ROOT/'summary.json');assert summary['status']=='completed' and summary['execution_valid'] and summary['source_integrity']
    assert summary['within_total_cap'] and summary['elapsed_seconds']<1800 and summary['restored_states']==8 and summary['model_definitions_constructed']==2
    assert read(ROOT/'attempt.json')=={'ordinal':1,'maximum_invocations':1}
    assert read(ROOT/'budget-start.json')['output_reserve_in_total_seconds']==60
    for k in ('training_updates','optimizer_proposals','random_parameter_initializations'):assert summary[k]==0
    weights=read(BASE/config['capacity_config'])['loss_weights'];a,b=weights['energy'],weights['force']
    totals={k:{'calls':0,'exposures':0} for k in ('prediction','gradient','finite_difference')}
    computed={}
    for case in config['cases']:
        name=case['name']; local=ROOT/name; original=SOURCE/case['width']; update=case['update']
        obs=read(local/'observations.json'); case_summary=read(local/'summary.json'); reported=report['cases'][name]
        assert case_summary['status']=='completed' and case_summary['verification_valid'] and obs['verification_valid'] and reported['endorsed']
        assert reported['observations']==obs and reported['summary']==case_summary
        prov=read(original/'provenance.json'); norm=prov['normalization'];h=read(original/f'histories/{update}.json')[-1]
        assert h['update']==update and obs['update']==update
        fit_ids=prov['inputs']['fitting']['record_ids'];anchor_ids=prov['inputs']['anchor']['record_ids']
        assert len(fit_ids)==len(set(fit_ids))==128 and len(anchor_ids)==16 and set(anchor_ids)<=set(fit_ids)
        with np.load(local/'predictions.npz',allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        computed[name]={}
        for role in ('fitting','anchor'):
            ids=prov['inputs'][role]['record_ids']; assert obs['record_ids'][role]==ids
            groups=[prov['inputs'][role]['memberships'][rid]['groups'][0] for rid in ids]
            with np.load(original/f'{role}-inputs.npz',allow_pickle=False) as ip:
                target=ip['energy_residual'].copy();ftarget=ip['forces'].copy();assert ip['positions'].shape[1]==21
            ep=[];fp=[];batchids=[]
            for batchfile in sorted((local/'prediction-batches').glob(role+'-*.npz')):
                meta=read(batchfile.with_suffix('.json'));assert meta['status']=='finite_valid' and meta['npz_sha256']==sha(batchfile)
                with np.load(batchfile,allow_pickle=False) as batch:
                    batchids+=batch['record_ids'].tolist();ep.append(batch['energy_prediction']);fp.append(batch['force_prediction'].reshape(-1,21,3))
                    start=len(batchids)-len(batch['record_ids']);stop=len(batchids)
                    assert np.array_equal(batch['energy_target'],target[start:stop]) and np.array_equal(batch['force_target'],ftarget[start:stop])
            assert batchids==ids and np.array_equal(np.concatenate(ep),arrays[role+'_energy']) and np.array_equal(np.concatenate(fp),arrays[role+'_forces'])
            e=arrays[role+'_energy']-target;f=arrays[role+'_forces']-ftarget
            assert np.array_equal(e,arrays[role+'_energy_error']);close(np.mean(f*f,axis=(1,2)),arrays[role+'_record_force_mse'])
            assert np.isfinite(e).all() and np.isfinite(f).all()
            values=pool(e,f);check_fields(obs['residuals'][role],values);computed[name][role]=values
            close(values['energy_mse_eV2'],values['mean_energy_error_squared_eV2']+values['centered_energy_error_variance_eV2'])
            close(values['energy_mse_eV2'],h['metrics'][role]['energy_mse_eV2']);close(values['force_mse_eV2_per_A2'],h['metrics'][role]['force_mse_eV2_per_A2'])
            per={}
            for group in sorted(set(groups)):
                ix=np.array([i for i,g in enumerate(groups) if g==group]); per[group]=pool(e[ix],f[ix]);check_fields(obs['residuals'][role]['per_group'][group],per[group])
                for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):close(per[group][key],h['metrics'][role]['per_group'][group][key])
            for key in ('energy_mse_eV2','force_mse_eV2_per_A2'):
                equal=np.mean([v[key] for v in per.values()]);close(equal,obs['residuals'][role]['equal_group'][key]);close(equal,h['metrics'][role]['equal_group'][key])
                close(values[key]/prov['original_baselines'][role][key],obs['original_baseline_fractions'][role][key])
            if update==6000:
                with np.load(original/'reference.npz',allow_pickle=False) as ref:
                    for suffix in ('energy','forces'):assert np.allclose(arrays[role+'_'+suffix],ref[role+'_'+suffix],atol=1e-10,rtol=0)
        env=serialization.msgpack_restore((original/f'snapshots/{update}.msgpack').read_bytes());state=serialization.msgpack_restore(env['payload'])
        assert hashlib.sha256(env['payload']).hexdigest()==env['sha256']
        assert env['provenance']==json.dumps(prov,sort_keys=True,separators=(',',':'))
        statehash=hashlib.sha256(serialization.to_bytes(tree(state))).hexdigest()
        assert obs['state_full_hash_before']==obs['state_full_hash_after']==statehash and obs['state_unchanged']
        assert state['update']==update and state['optimizer']['0']['count']==update and json.loads(state['normalization_json'])==norm
        receipt={'update':update,'checkpoint':f'snapshots/{update}.msgpack','checkpoint_sha256':sha(original/f'snapshots/{update}.msgpack'),'state_hashes':h['state_hashes']}
        audit=cc.audit_state(original,receipt,prov);assert audit['state_hashes']==obs['state_hashes_before']==obs['state_hashes_after']
        assert read(local/'source-checkpoint-audit.json')=={'sha256':receipt['checkpoint_sha256'],'unchanged':True}
        grad=serialization.msgpack_restore((local/'gradients.msgpack').read_bytes());ge=grad['energy_scaled'];gf=grad['force_scaled']
        for label,c,d in [('joint_unweighted',1.,1.),('joint_weighted',a,b)]:
            actual=leaves(grad[label]);el=leaves(ge);fl=leaves(gf);err=max(float(np.max(abs(actual[k]-c*el[k]-d*fl[k]))) for k in el)
            for k in el:close(actual[k],c*el[k]+d*fl[k])
            observed=obs['gradients']['unweighted_joint_identity' if label=='joint_unweighted' else 'weighted_joint_identity'];assert observed['valid'];close(err,observed['maximum_absolute_error'])
        for scale,eg,fg in [('unweighted',ge,gf),('weighted',tree(ge),tree(gf))]:
            def multiply(v,c):return {k:multiply(x,c) for k,x in v.items()} if isinstance(v,dict) else c*v
            if scale=='weighted':eg=multiply(eg,a);fg=multiply(fg,b)
            scopes={'whole':(eg,fg),'backbone':(eg['Backbone_0'],fg['Backbone_0']),'head_kernel':(eg['energy_head']['0+']['kernel'],fg['energy_head']['0+']['kernel']),'head_bias':(eg['energy_head']['0+']['bias'],fg['energy_head']['0+']['bias'])}
            for label,(x,y) in scopes.items():check_fields(obs['gradients'][scale][label],geometry(x,y,a,b,scale=='weighted'))
            el=leaves(eg);fl=leaves(fg)
            for key in el:check_fields(obs['gradients'][scale]['leaves'][key],geometry(el[key],fl[key],a,b,scale=='weighted'))
        expected=2*21*computed[name]['fitting']['signed_mean_energy_error_eV']/norm['energy_loss_scale_eV']**2
        bias=obs['bias_check'];close(expected,bias['expected_unweighted_energy']);close(expected,float(ge['energy_head']['0+']['bias'][0]));close(a*expected,bias['expected_weighted_energy']);assert abs(gf['energy_head']['0+']['bias'][0])<=1e-12 and bias['valid']
        losses=obs['losses'];close(losses['energy_scaled'],computed[name]['fitting']['energy_mse_eV2']/norm['energy_loss_scale_eV']**2);close(losses['force_scaled'],computed[name]['fitting']['force_mse_eV2_per_A2']/norm['force_loss_scale_eV_per_A']**2);close(losses['unweighted_joint'],losses['energy_scaled']+losses['force_scaled']);close(losses['weighted_joint'],a*losses['energy_scaled']+b*losses['force_scaled'])
        for objective,g in [('energy',ge),('force',gf)]:
            fd=obs['finite_differences'][objective];gnorm=sum(float(np.vdot(v,v)) for v in leaves(g).values())**.5;assert fd['direction_kind']=='unit_gradient';close(fd['autodiff_directional_derivative'],gnorm)
            assert [row['step'] for row in fd['steps']]==[1e-4,1e-5]
            for row in fd['steps']:
                close(row['absolute_error'],abs(row['finite_difference']-fd['autodiff_directional_derivative']))
                assert row['valid']==bool(np.isclose(row['finite_difference'],fd['autodiff_directional_derivative'],atol=1e-8,rtol=1e-4))
                if not row['valid']:result['failed_fd_steps'].append({'case':name,'objective':objective,**row})
            assert fd['valid']==any(row['valid'] for row in fd['steps'])
        work=read(local/'work.json');assert work['in_flight'] is None
        for kind,calls,exposures in [('prediction',9,144),('gradient',32,512),('finite_difference',64,1024)]:
            row=work['counts'][kind];assert row['attempted_calls']==row['returned_calls']==row['valid_calls']==calls;assert row['returned_structure_exposures']==row['valid_structure_exposures']==exposures
            totals[kind]['calls']+=calls;totals[kind]['exposures']+=exposures
        for k in ('training_updates','optimizer_proposals','random_parameter_initializations'):assert work[k]==0
        result['cases'][name]={'status':'verified','fitting':computed[name]['fitting'],'endpoint_reference_checked':update==6000,'work_unknown':False}
    for width,pre,peak,nextu in [('width32',3600,3700,3800),('width64',4900,5000,5100)]:
        for role in ('fitting','anchor'):
            for label,x,y in [('pre_to_peak',pre,peak),('peak_to_next',peak,nextu)]:
                before=computed[f'{width}-u{x}'][role];after=computed[f'{width}-u{y}'][role];delta=after['energy_mse_eV2']-before['energy_mse_eV2'];m=after['mean_energy_error_squared_eV2']-before['mean_energy_error_squared_eV2'];v=after['centered_energy_error_variance_eV2']-before['centered_energy_error_variance_eV2']
                check_fields(report['contrasts'][width][role][label],{'energy_mse_change_eV2':delta,'mean_squared_change_eV2':m,'variance_change_eV2':v,'offset_increase_share':m/delta if delta>0 else None});close(delta,m+v)
    assert len(result['failed_fd_steps'])==2
    with zipfile.ZipFile(RECORDS/'numerical-run.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==247 and set(z.namelist())==set(raw_map)
        for n,s in raw_map.items():assert hashlib.sha256(z.read(n)).hexdigest()==s
    spec=importlib.util.spec_from_file_location('review_saved_report',BASE/'scripts/diagnose_capacity_spikes.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='spike-independent-report-') as tmp:
        actual=module.render(ROOT,Path(tmp).resolve());assert actual==report
        for name in ('capacity-spike-diagnostics.json','capacity-spike-diagnostics.md'):assert (Path(tmp)/name).read_bytes()==(BASE/'reports'/name).read_bytes()
    inventory(SOURCE,source_map);inventory(ROOT,raw_map)
    result.update(status='PASS',source_raw_files=343,source_pins=347,diagnostic_raw_files=247,archive_crc='passed',all_archive_bytes_match=True,source_and_raw_unchanged=True,work_totals=totals,report_sha256={name:sha(BASE/'reports'/name) for name in ('capacity-spike-diagnostics.json','capacity-spike-diagnostics.md')},archive_sha256=sha(RECORDS/'numerical-run.zip'),analysis_script_sha256=code['script_sha256'],elapsed_numerical_seconds=summary['elapsed_seconds'],fd_scope='Saved directional derivatives, arithmetic and criteria audited; central objective evaluations not repeated')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        (OUT/'attempt1-failure.json').write_text(json.dumps({'status':'failed_inspection','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
