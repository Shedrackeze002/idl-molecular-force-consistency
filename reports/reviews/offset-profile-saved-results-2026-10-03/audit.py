"""Independent saved-only checks; never invokes a numerical model."""
import hashlib, json, sys, time, zipfile, subprocess
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); OUT=Path(__file__).parent
RUN=ROOT/'runs/aspirin-offset-profile-2026-10-02'
OLD=ROOT/'runs/aspirin-capacity-2026-10-02/width32'
EX=ROOT/'reports/execution/offset-profile-2026-10-02'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b): assert np.isclose(a,b,atol=1e-14,rtol=1e-11),(a,b)
def score(e,f,ix):
    e=e[ix]; f=f[ix]
    d={'energy_mse_eV2':float(np.mean(e*e)), 'force_mse_eV2_per_A2':float(np.mean(f*f)),
       'signed_mean_eV':float(np.mean(e)), 'squared_mean_eV2':float(np.mean(e)**2),
       'centered_population_variance_eV2':float(np.mean((e-np.mean(e))**2))}
    near(d['energy_mse_eV2'],d['squared_mean_eV2']+d['centered_population_variance_eV2'])
    return d
started=time.monotonic()
report=read(ROOT/'reports/offset-profile-diagnostics.json'); p=read(RUN/'provenance.json'); original=read(OLD/'provenance.json')
assert sha(ROOT/'reports/offset-profile-paper-findings.md')=='9de7272b493048adb50ab8e86e7fb28097afd0da29e8fbafba5c48277e7e1d66'
assert p['inputs']==original['inputs']
arithmetic={'roles':{},'spike_components':{},'matched_grid':{}}
with np.load(OLD/'reference.npz',allow_pickle=False) as a,np.load(RUN/'predictions/6000.npz',allow_pickle=False) as b:
    for role in ('fitting','anchor'):
        meta=p['inputs'][role];ids=meta['record_ids'];members=meta['memberships']
        assert len(set(ids))==len(ids) and all(len(members[i]['groups'])==1 for i in ids)
        assert sha(RUN/(role+'-inputs.npz'))==sha(OLD/(role+'-inputs.npz'))
        groups=sorted({members[i]['groups'][0] for i in ids}); arm={}
        with np.load(RUN/(role+'-inputs.npz'),allow_pickle=False) as targets:
            for name,arrays in [('historical',a),('intervention',b)]:
                e=arrays[role+'_energy']-targets['energy_residual']; f=arrays[role+'_forces']-targets['forces']
                pool=score(e,f,list(range(len(ids)))); gg={g:score(e,f,[j for j,i in enumerate(ids) if members[i]['groups']==[g]]) for g in groups}
                counts={g:sum(members[i]['groups']==[g] for i in ids) for g in groups}
                within=sum(counts[g]*gg[g]['centered_population_variance_eV2']/len(ids) for g in groups)
                between=sum(counts[g]*(gg[g]['signed_mean_eV']-pool['signed_mean_eV'])**2/len(ids) for g in groups)
                near(pool['centered_population_variance_eV2'],within+between)
                arm[name]={'pooled':pool,'per_group':gg,'within':within,'between':between}
        keys=['energy_mse_eV2','force_mse_eV2_per_A2','centered_population_variance_eV2']
        changes={k:arm['intervention']['pooled'][k]/arm['historical']['pooled'][k]-1 for k in keys}
        for k in keys[:2]:
            contrast=report['contrasts'][role][k]
            near(arm['historical']['pooled'][k],contrast['historical_endpoint']);near(arm['intervention']['pooled'][k],contrast['intervention_endpoint'])
            near(changes[k],contrast['relative_endpoint_change'])
        fractions={k:arm['intervention']['pooled'][k]/original['original_baselines'][role][k] for k in keys[:2]}
        assert all(v>0.1 for v in fractions.values()) and report['gates'][role]['accepted'] is False
        for k,v in fractions.items():near(v,report['gates'][role]['fraction_remaining'][k])
        gc={k:{'lower':sum(arm['intervention']['per_group'][g][k]<arm['historical']['per_group'][g][k] for g in groups),
               'higher':sum(arm['intervention']['per_group'][g][k]>arm['historical']['per_group'][g][k] for g in groups)} for k in keys}
        arithmetic['roles'][role]={'count':len(ids),'groups':groups,'group_counts':counts,'arms':arm,'relative_changes':changes,'fractions':fractions,'group_changes':gc}
assert set(p['inputs']['anchor']['record_ids'])<=set(p['inputs']['fitting']['record_ids'])
assert arithmetic['roles']['anchor']['groups']==['01']
hist=read(RUN/'histories/6000.json'); oldhist=read(OLD/'histories/6000.json')
updates=sorted(({r['update'] for r in hist}&{r['update'] for r in oldhist})-{0})
assert len(updates)==62 and 3001 in updates
for role in ('fitting','anchor'):
    arithmetic['matched_grid'][role]={}
    for k in ('energy_mse_eV2','force_mse_eV2_per_A2'):
        maxima={name:max(r['metrics'][role][k] for r in rows if r['update'] in updates) for name,rows in [('historical',oldhist),('intervention',hist)]}
        for name,v in maxima.items():near(v,report['contrasts'][role][k][name+'_matching_grid_maximum'])
        arithmetic['matched_grid'][role][k]=maxima
arithmetic['matched_updates']=updates
component=read(ROOT/'reports/offset-profile-component-comparison.json')
mapping={'energy_mse_eV2':'energy_mse_eV2','force_mse_eV2_per_A2':'force_mse_eV2_per_A2','signed_mean_energy_error_eV':'signed_mean_eV','mean_energy_error_squared_eV2':'squared_mean_eV2','centered_energy_error_variance_eV2':'centered_population_variance_eV2'}
for u in (3600,3700,3800,6000):
    spike=ROOT/f'runs/aspirin-capacity-spike-2026-10-02/width32-u{u}'
    observation=read(spike/'observations.json'); assert observation['verification_valid']
    arithmetic['spike_components'][str(u)]={}
    with np.load(spike/'predictions.npz',allow_pickle=False) as a,np.load(RUN/f'predictions/{u}.npz',allow_pickle=False) as b:
        for role in ('fitting','anchor'):
            ids=p['inputs'][role]['record_ids'];assert observation['record_ids'][role]==ids
            with np.load(RUN/(role+'-inputs.npz'),allow_pickle=False) as target:
                for name,arrays in [('historical',a),('intervention',b)]:
                    e=arrays[role+'_energy']-target['energy_residual'];f=arrays[role+'_forces']-target['forces']
                    result=score(e,f,list(range(len(ids))))
                    for source,targetkey in mapping.items():near(result[targetkey],component['comparisons'][str(u)][role]['pooled'][source][name])
                    for g,v in component['comparisons'][str(u)][role]['same_role_groups'].items():
                        gg=score(e,f,[j for j,i in enumerate(ids) if g in p['inputs'][role]['memberships'][i]['groups']])
                        for source,targetkey in mapping.items():near(gg[targetkey],v[source][name])
            arithmetic['spike_components'][str(u)][role]='raw arrays and all pooled/group component values agree'
# Independent raw inventory and exact member checks (ZIP read validates CRC).
expected=read(EX/'numerical-run-inventory.json');actual={str(f.relative_to(RUN)):sha(f) for f in RUN.rglob('*') if f.is_file()}
assert actual==expected and len(actual)==18324
with zipfile.ZipFile(EX/'numerical-run.zip') as z:
    names=z.namelist();assert len(names)==len(set(names))==18324 and set(names)==set(expected)
    for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==expected[n],n
    assert z.testzip() is None
preserved=read(EX/'prelaunch-source-preservation.json');prefix=(ROOT/'reports/PAPER_EVIDENCE_LOG.md').read_bytes()[:preserved['protected_evidence_prefix_bytes']]
assert hashlib.sha256(prefix).hexdigest()==preserved['protected_evidence_prefix_sha256']
gate=read(EX/'prelaunch-gate.json'); pins=read(EX/'source-pins.json')
for path,h in {**pins,**gate['artifacts'],**gate['code']['code_files']}.items():assert sha(ROOT/path)==h,path
assert len(pins)==601 and gate['status']=='cleared' and gate['run_ordinal']==1
assert gate['launch_revision']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert (EX/'fullsuite-v2.txt').read_text().strip().endswith('656 passed in 488.59s (0:08:08)')
inv=read(EX/'production-invocation.json');assert inv['ordinal']==1 and inv['gate']==gate
process=read(EX/'production-process-completion.json');assert process['exit_code']==0
assert sha(EX/'production-stderr.txt')==process['stderr_sha256'] and sha(EX/'production-stdout.txt')==process['stdout_sha256']
summary=read(RUN/'summary.json');assert summary['execution_valid'] and summary['final_durable_write_checked'] and summary['elapsed_seconds']<3600
assert summary['updates']==6000 and summary['accepted_calibrations']==6001 and summary['random_initializations']==0
assert summary['pending_inflight_unknown'] is None and not summary['late_call'] and not summary['pending_update_not_accepted']
child=read(RUN/'checkpoint-verification.json'); assert child['maximum_absolute_error']==0 and child['prediction_frames']==144
assert child['training_updates']==0 and child['random_initializations']==0 and child['adam_count']==6000
assert read(RUN/'probe-attempt.json')['ordinal']==1 and read(RUN/'probe-process.json')['exit_code']==0
for entry in summary['work'].values():
    assert entry['attempted_calls']==entry['returned_calls']==entry['valid_calls']
    assert entry['returned_frames']==entry['valid_frames']
assert summary['work']['training:gradient']['valid_calls']==48000
assert summary['work']['training:calibration_energy']['valid_calls']==48000
assert summary['work']['training:adam_proposal']['valid_calls']==6000
with np.load(RUN/'predictions/6000.npz',allow_pickle=False) as arrays:
    for role in ('fitting','anchor'):
        with np.load(RUN/f'staging/child-{role}-returned.npz',allow_pickle=False) as returned:
            assert np.array_equal(returned['energy'],arrays[role+'_energy'][-16:])
            assert np.array_equal(returned['forces'].reshape(16,21,3),arrays[role+'_forces'][-16:])
# Reconstruct the canonical report using only saved arrays/state audit. This checks
# all 6000 accepted links, 6001 projection vectors, all scores, oracle tensors,
# child binding and frozen runtime attribution without model evaluation.
sys.path.insert(0,str(ROOT/'scripts'))
import report_offset_profile as renderer
regenerated=renderer.render(RUN,OUT/'regenerated')
assert regenerated['execution_valid'],regenerated.get('reason')
for ext in ('json','md'):assert sha(OUT/'regenerated'/('offset-profile-diagnostics.'+ext))==sha(ROOT/'reports'/('offset-profile-diagnostics.'+ext))
oracle_evidence={}
for u in (1,3001):
    import msgpack
    o=read(RUN/f'oracles/{u}.json')
    d=renderer.decoded(msgpack.unpackb((RUN/f'oracles/{u}-oracle_complete.msgpack').read_bytes(),raw=False))
    state=renderer.decoded(renderer.cc._envelope(RUN/f'snapshots/{u}.msgpack',p))
    def differences(a,b):
        if isinstance(a,dict):return [x for k in a for x in differences(a[k],b[k])]
        a=np.asarray(a);b=np.asarray(b);err=np.abs(a-b);bound=o['atol']+o['rtol']*np.abs(b)
        assert np.all(err<=bound)
        return [float(np.max(err))]
    gradient=max(differences(d['reference_gradient'],d['actual_gradient']))
    state_error=max(differences(d['reference_params'],state['params'])+differences(d['reference_optimizer'],state['optimizer']))
    prediction=max(differences(d['reference_energy_adjusted'],d['actual_energy'])+differences(d['reference_forces'],d['actual_forces']))
    near(gradient,o['gradient_maximum_error']);near(state_error,o['state_maximum_error']);near(prediction,o['prediction_maximum_error'])
    oracle_evidence[str(u)]={'combined_per_leaf_tolerance_passed':True,'gradient_maximum_error':gradient,'state_maximum_error':state_error,'prediction_maximum_error':prediction,'incoming_nonzero_moments':o['incoming_nonzero_moments']}
assert {str(f.relative_to(RUN)):sha(f) for f in RUN.rglob('*') if f.is_file()}==actual
result={'status':'PASS','saved_only':True,'new_model_calls':0,'new_training_updates':0,'new_predictions':0,
        'raw_files':len(actual),'archive_members':len(names),'all_member_hashes_and_crc':'PASS','source_pins':len(pins),
        'frozen_participant_hashes':'PASS','protected_log_prefix':'PASS','canonical_regeneration':'byte-for-byte',
        'elapsed_saved_review_seconds':time.monotonic()-started,'arithmetic':arithmetic,
        'oracles':oracle_evidence,
        'execution':{k:summary[k] for k in ('updates','accepted_calibrations','elapsed_seconds','final_durable_write_checked','random_initializations')},
        'reviewed_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [ROOT/'reports/offset-profile-paper-findings.md',ROOT/'reports/offset-profile-diagnostics.json',ROOT/'reports/offset-profile-diagnostics.md',ROOT/'reports/offset-profile-component-comparison.json',EX/'saved-closeout-verification.json',EX/'independent-arithmetic.json',EX/'numerical-run.zip',EX/'numerical-run-inventory.json',EX/'prelaunch-gate.json',EX/'source-pins.json',RUN/'summary.json',RUN/'predictions/6000.npz']}}
(OUT/'audit-results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('arithmetic','reviewed_sha256')},indent=2))
