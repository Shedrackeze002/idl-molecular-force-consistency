"""Independent endpoint arithmetic from raw NPZ bytes, no model imports/calls."""
import hashlib,json
from pathlib import Path
import numpy as np
p=Path(__file__).parent;old=Path('runs/aspirin-capacity-2026-10-02/width32');new=Path('runs/aspirin-offset-profile-2026-10-02')
original=json.loads((old/'provenance.json').read_text());current=json.loads((new/'provenance.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
result={'status':'raw_saved_endpoint_arithmetic_verified','new_model_calls':0,'roles':{},'files_sha256':{}}
with np.load(old/'reference.npz',allow_pickle=False) as a,np.load(new/'predictions/6000.npz',allow_pickle=False) as b:
 for role in ('fitting','anchor'):
  assert original['inputs'][role]==current['inputs'][role]
  assert sha(old/(role+'-inputs.npz'))==sha(new/(role+'-inputs.npz'))
  with np.load(new/(role+'-inputs.npz'),allow_pickle=False) as data:
   ids=current['inputs'][role]['record_ids'];members=current['inputs'][role]['memberships']
   groups=sorted({g for r in ids for g in members[r]['groups']})
   assert all(len(members[r]['groups'])==1 for r in ids)
   paired={}
   for arm,pred in [('historical',a),('intervention',b)]:
    e=pred[role+'_energy']-data['energy_residual'];f=pred[role+'_forces']-data['forces']
    def score(ii):
     ee=e[ii];ff=f[ii]
     return {'energy_mse_eV2':float(np.mean(ee**2)),'force_mse_eV2_per_A2':float(np.mean(ff**2)),
      'mean_eV':float(np.mean(ee)),'mean_squared_eV2':float(np.mean(ee)**2),'centered_variance_eV2':float(np.var(ee))}
    per={g:score([i for i,r in enumerate(ids) if g in members[r]['groups']]) for g in groups};whole=score(list(range(len(ids))))
    within=sum(len(per[g] and [r for r in ids if g in members[r]['groups']])/len(ids)*per[g]['centered_variance_eV2'] for g in groups)
    between=sum(len([r for r in ids if g in members[r]['groups']])/len(ids)*(per[g]['mean_eV']-whole['mean_eV'])**2 for g in groups)
    assert np.isclose(whole['energy_mse_eV2'],whole['mean_squared_eV2']+whole['centered_variance_eV2'],atol=1e-14,rtol=1e-12)
    assert np.isclose(whole['centered_variance_eV2'],within+between,atol=1e-14,rtol=1e-12)
    paired[arm]={'pooled':whole,'per_group':per,'within_group_variance_eV2':within,'between_group_variance_eV2':between}
   rel={k:paired['intervention']['pooled'][k]/paired['historical']['pooled'][k]-1 for k in ('energy_mse_eV2','force_mse_eV2_per_A2','centered_variance_eV2')}
   groupcounts={k:{'lower':sum(paired['intervention']['per_group'][g][k]<paired['historical']['per_group'][g][k] for g in groups),
     'higher':sum(paired['intervention']['per_group'][g][k]>paired['historical']['per_group'][g][k] for g in groups)} for k in ('energy_mse_eV2','force_mse_eV2_per_A2','centered_variance_eV2')}
   result['roles'][role]={'record_count':len(ids),'groups':groups,'paired':paired,'relative_endpoint_changes':rel,'group_change_counts':groupcounts,
     'original_baseline_fractions':{k:paired['intervention']['pooled'][k]/original['original_baselines'][role][k] for k in ('energy_mse_eV2','force_mse_eV2_per_A2')}}
for path in [old/'reference.npz',new/'predictions/6000.npz',old/'provenance.json',new/'provenance.json',new/'fitting-inputs.npz',new/'anchor-inputs.npz']:result['files_sha256'][str(path)]=sha(path)
with (p/'independent-arithmetic.json').open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({role:{k:result['roles'][role][k] for k in ['groups','relative_endpoint_changes','group_change_counts']} for role in result['roles']},indent=2))
