"""Once-only offset profiling; public source/gate guards precede JAX imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
from force_consistency import capacity_contract as cc

CONFIG_ID='4cae0f4ac8703459bd8a9193b27eb2b287365c4bbc936658fa0653ce37421cf3'
EXTRAS=('configs/aspirin-capacity-spike.json','reports/capacity-spike-diagnostics.json',
 'reports/capacity-spike-diagnostics.md','reports/capacity-spike-paper-findings.md',
 'docs/reviews/2026-10-02-capacity-spike-saved-results.md',
 'reports/execution/spike-diagnostic-2026-10-02/final-closeout.json')

def verify_inputs(c):
    if cc.canonical_hash(c)!=CONFIG_ID:raise ValueError('Unreviewed offset configuration')
    for key in ('source_pins','base_pins','spike_inventory','plan_gate'):
        if cc.digest(cc.no_symlinks(c[key]))!=c[key+'_sha256']:raise ValueError('Source manifest/review gate changed: '+key)
    plan=cc.read(c['plan_gate'])
    if plan['status']!='two_final_plan_reviews_PASS':raise ValueError('Two final plan reviews missing')
    cc.verify_pins(plan['artifacts'])
    pins=cc.read(c['source_pins']);base=cc.read(c['base_pins']);inventory=cc.read(c['spike_inventory'])
    expected=set(base)|{c['base_pins']}|{'runs/aspirin-capacity-spike-2026-10-02/'+n for n in inventory}|set(EXTRAS)
    if len(pins)!=601 or set(pins)!=expected:raise ValueError('Incomplete exact source membership')
    if any(pins[n]!=sha for n,sha in base.items()) or any(pins['runs/aspirin-capacity-spike-2026-10-02/'+n]!=s for n,s in inventory.items()):
        raise ValueError('Inventory/map disagrees with source pins')
    cc.verify_pins(pins)
    original=cc.read('configs/aspirin-capacity.json');cc.verify_sources(original)
    root=cc.no_symlinks(c['source_root']);p=cc.read(root/'provenance.json');s=cc.read(root/'summary.json')
    receipt=cc.read(root/'commit.json');hist=cc.read(root/receipt['history']);child=cc.read(root/'checkpoint-verification.json')
    if (not s['execution_valid'] or s['status']!='completed' or receipt['update']!=6000
        or s['history']!=hist or p['original_baselines']!=hist[0]['metrics']
        or p['model']!={**c['model'],'features':32} or p['normalization']!=json.loads(cc._envelope(root/'snapshots/0.msgpack',p)['normalization_json'])
        or p['case']['name']!='width32' or child['status']!='passed' or child['maximum_absolute_error']>1e-10
        or len(p['inputs']['fitting']['record_ids'])!=128 or len(p['inputs']['anchor']['record_ids'])!=16
        or not set(p['inputs']['anchor']['record_ids'])<=set(p['inputs']['fitting']['record_ids'])):
        raise ValueError('Historical reference binding differs')
    zero_receipt={'update':0,'checkpoint':'snapshots/0.msgpack','checkpoint_sha256':cc.digest(root/'snapshots/0.msgpack'),'state_hashes':p['initial_state_hashes']}
    zero=cc.audit_state(root,zero_receipt,p);cc.audit_state(root,receipt,p);cc.verify_code(p['code'])
    if zero['parameter_count']!=13569 or zero['params_schema']['energy_head']['0+']['bias']!={'shape':[1],'dtype':'float64'}:
        raise ValueError('Source blueprint differs')
    return {'root':str(root),'provenance':p,'pins':pins,'zero':zero,'zero_receipt':zero_receipt}

def verify_gate(c,expected=None):
    gate=cc.read(cc.no_symlinks(c['prelaunch_gate']))
    if gate.get('status')!='cleared' or (expected is not None and gate!=expected):raise ValueError('Frozen prelaunch gate changed')
    cc.verify_pins(gate['artifacts'])
    return gate

def validate_launch_code(code,provenance,gate):
    if provenance.get('code')!=code:raise ValueError('Local code/provenance differs')
    if gate['status']=='cleared':
        cc.verify_code(code)
        if code['code_fingerprint']!=gate['code_fingerprint'] or code['revision']!=gate['launch_revision'] or gate.get('code',code)!=code:
            raise ValueError('Launch code differs from frozen gate')

def verify_invocation(c,gate):
    invocation=cc.read(Path(c['prelaunch_gate']).parent/'production-invocation.json')
    if (invocation['ordinal']!=1 or invocation['gate']!=gate or invocation['config_identity']!=cc.canonical_hash(c)
        or invocation['run_directory']!=c['run_directory']):raise ValueError('Worker differs from supervisor frozen invocation')

def finalize_summary(j,out):
    """Confirm the clock after durable writes; correction can never restore validity."""
    j.save()
    def measured():
        out['elapsed_seconds']=j.clock()-j.start
        if out['elapsed_seconds']>=j.deadline-j.start:
            out.update(status='incomplete',execution_valid=False,stop_reason='wall_budget')
    measured();j.owner.json('summary.json',out)
    measured();out['final_durable_write_checked']=True;j.owner.json('summary.json',out)
    # This boundary includes the elapsed-record write itself.
    if j.clock()>=j.deadline:
        measured();j.owner.json('summary.json',out)
        out['elapsed_seconds']=j.clock()-j.start
    return out

def reserve_root(root):
    root=cc.no_symlinks(root);root.mkdir(exist_ok=False)
    owner=cc.DirectoryLease(root)
    owner.json('attempt.json',{'ordinal':1,'started_at_utc':cc.utc(),'directory_identity':list(owner.identity),'no_retry':True},exclusive=True)
    return owner

class Journal:
    def __init__(self,owner,wall,reserve,*,clock=time.perf_counter,start=None,filename='work.json'):
        self.owner=owner;self.clock=clock;self.start=clock() if start is None else start
        self.deadline=self.start+wall;self.reserve=reserve;self.filename=filename
        self.work={};self.pending=None;self.late_call=False;self.phase='setup';self.update=0;self.accepted=0
    def check(self,final=False):
        self.owner.check()
        if self.clock()+(0 if final else self.reserve)>=self.deadline:raise TimeoutError('Total wall cap/reserve exhausted')
    def save(self):
        self.owner.json(self.filename,{'work':self.work,'pending_inflight_unknown':self.pending,
            'late_call':self.late_call,'phase':self.phase,'outgoing_update':self.update,
            'accepted_updates':self.accepted,'elapsed_seconds':self.clock()-self.start})
    def call(self,kind,count,fn,*args,on_returned=lambda x:None):
        self.check();key=self.phase+':'+kind if self.phase!='setup' else kind
        w=self.work.setdefault(key,{'attempted_calls':0,'returned_calls':0,'valid_calls':0,'returned_frames':0,'valid_frames':0})
        w['attempted_calls']+=1;self.pending={'kind':key,'frames':count,'update':self.update};self.save()
        result=fn(*args)
        import jax
        jax.block_until_ready(result)
        w['returned_calls']+=1;w['returned_frames']+=count;self.pending=None;self.save()
        on_returned(result)
        from force_consistency.runs import assert_finite
        assert_finite(result)
        w['valid_calls']+=1;w['valid_frames']+=count;self.save()
        try:self.check()
        except TimeoutError:self.late_call=True;self.save();raise
        return result

class Store:
    """Bound child leases. Binary returned data precedes strict finite JSON."""
    def __init__(self,owner):self.owner=owner;self.children={}
    def child(self,name):
        if name not in self.children:self.children[name]=self.owner.child(name,create=not (self.owner.path/name).exists())
        return self.children[name]
    def write(self,group,name,data,exclusive=False):self.child(group).write(name,data,exclusive=exclusive)
    def json(self,group,name,value,exclusive=False):self.child(group).json(name,value,exclusive=exclusive)
    def npz(self,group,name,values,exclusive=False):
        import numpy as np
        stream=io.BytesIO();np.savez(stream,**values);self.write(group,name,stream.getvalue(),exclusive)
        return cc.digest(self.owner.path/group/name)
    def close(self):
        for child in self.children.values():child.close()

def state_fields_equal(state,zero):
    import numpy as np
    if (state['sampler_rng_json']!=zero['sampler_rng_json'] or state['normalization_json']!=zero['normalization_json']
        or not np.array_equal(state['model_rng'],zero['model_rng'])):raise ValueError('Source RNG/normalization fields changed')

def projection_binding(prov,gate,phase,incoming,outgoing):
    return {'ordinal':1,'phase':phase,'incoming_update':incoming,'outgoing_update':outgoing,
        'fitting_record_ids':prov['inputs']['fitting']['record_ids'],
        'fitting_input_sha256':prov['input_files_sha256']['fitting'],
        'config_identity':prov['config_identity'],'gate_identity':cc.canonical_hash(gate)}

def predictions(state,parts,model,micro,journal,store,label):
    import numpy as np
    from force_consistency import late_rate as lr
    from force_consistency.coverage_runs import _predict
    result={}
    for role,p in parts.items():
        ee=[];ff=[]
        for first in range(0,len(p.record_ids),micro):
            idx=np.arange(first,min(first+micro,len(p.record_ids)));b=lr.partition_batch(p,idx)
            e,f=journal.call('prediction_'+role,len(idx),_predict(model),state['params'],b,
                on_returned=lambda a:store.npz('staging',label+'-'+role+'-returned.npz',{'energy':a[0],'forces':a[1]}))
            ee.append(np.asarray(e));ff.append(np.asarray(f).reshape(len(idx),*p.forces.shape[1:]))
        result[role+'_energy']=np.concatenate(ee);result[role+'_forces']=np.concatenate(ff)
    return result

def metrics(parts,arrays):
    import numpy as np
    from force_consistency.coverage_runs import _group_score
    result={}
    for role,p in parts.items():
        e=arrays[role+'_energy'];f=arrays[role+'_forces'];r=e-p.energy_residual
        score=_group_score(p,e,f)
        def decomposition(ids):
            indices=[list(p.record_ids).index(x) for x in ids];v=r[indices]
            return {'signed_mean_eV':float(np.mean(v)),'squared_mean_eV2':float(np.mean(v)**2),'centered_population_variance_eV2':float(np.var(v))}
        score.update(decomposition(p.record_ids))
        for value in score['per_group'].values():value.update(decomposition(value['record_ids']))
        result[role]=score
    return result

def probe_binding(root):
    root=cc.no_symlinks(root);receipt=cc.read(root/'commit.json')
    u=receipt['update']
    names=['config.json','code.json','provenance.json','prelaunch-gate.json','source-pins.json',
        'shared-zero.msgpack','shared-zero-provenance.json','snapshots/0.msgpack','histories/0.json',
        'commit.json',receipt['checkpoint'],receipt['history'],receipt['predictions'],
        'fitting-inputs.npz','anchor-inputs.npz',f'accepted/{u}.json',f'projections/training-{u}.json',f'projections/training-{u}.npz']
    stat=root.stat()
    return {'root':str(root.absolute()),'directory_identity':[stat.st_dev,stat.st_ino],
        'files_sha256':{n:cc.digest(cc.no_symlinks(root/n)) for n in set(names)}}

def reserve_probe(root,deadline):
    owner=cc.DirectoryLease(root)
    try:
        attempt={'ordinal':1,'token':secrets.token_hex(32),'deadline_monotonic':deadline,'binding':probe_binding(root)}
        owner.json('probe-attempt.json',attempt,exclusive=True);return attempt
    finally:owner.close()

def claim_probe_entry(root,token):
    root=Path(root).absolute();attempt=cc.read(root/'probe-attempt.json')
    if attempt['ordinal']!=1 or attempt['token']!=token or attempt['binding']!=probe_binding(root):raise ValueError('Child token/binding mismatch')
    owner=cc.DirectoryLease(root,expected=attempt['binding']['directory_identity'])
    try:
        entry={'attempt_sha256':cc.digest(root/'probe-attempt.json'),'binding':attempt['binding'],'entered_at_utc':cc.utc()}
        owner.json('probe-entry.json',entry,exclusive=True)
        return entry
    finally:owner.close()

def launch_probe(root,timeout,token):
    if timeout<=0:raise TimeoutError('No child budget remains')
    owner=cc.DirectoryLease(root)
    try:
        try:r=subprocess.run([sys.executable,str(Path(__file__).absolute()),'--probe',str(root),'--token',token],capture_output=True,timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            owner.write('probe-stdout.txt',exc.stdout or b'');owner.write('probe-stderr.txt',exc.stderr or b'');raise
        owner.write('probe-stdout.txt',r.stdout);owner.write('probe-stderr.txt',r.stderr)
        owner.json('probe-process.json',{'exit_code':r.returncode,'timeout_seconds':timeout})
        if r.returncode:raise RuntimeError('Fresh process child failed: '+r.stderr.decode(errors='replace')[-2000:])
        verification=cc.read(root/'checkpoint-verification.json');completion=cc.read(root/'probe-completion.json')
        if completion['status']!='passed' or completion['verification_sha256']!=cc.digest(root/'checkpoint-verification.json'):
            raise ValueError('Missing valid child completion')
        return verification
    finally:owner.close()

def probe(root,token):
    root=Path(root).absolute();c=cc.read(root/'config.json');source=verify_inputs(c);gate=verify_gate(c)
    if root!=(Path.cwd()/c['run_directory']).absolute() or gate!=cc.read(root/'prelaunch-gate.json'):raise ValueError('Unreviewed child root/gate')
    cc.verify_code(cc.read(root/'code.json'));attempt=cc.read(root/'probe-attempt.json')
    if cc.read(root/'commit.json')['update']!=c['max_updates']:raise ValueError('Probe requires reviewed endpoint')
    entry=claim_probe_entry(root,token)
    return probe_owned(root,source,gate,attempt,entry)

def probe_owned(root,source,gate,attempt,entry):
    """Numerical child after claimed entry; private fixture uses this same boundary."""
    c=cc.read(root/'config.json')
    owner=cc.DirectoryLease(root,expected=attempt['binding']['directory_identity']);store=Store(owner)
    try:
        j=Journal(owner,attempt['deadline_monotonic']-time.perf_counter(),0,filename='probe-work.json');j.phase='child';j.check()
        from report_offset_profile import audit_saved
        audit_saved(root,complete=False,source=source,gate=gate)
        import jax
        import numpy as np
        from force_consistency import late_rate as lr
        from offset_profile_core import restore_state,validate_bias_moments,full_hash
        p=cc.read(root/'provenance.json');receipt=cc.read(root/'commit.json')
        with jax.default_device(lr.require_backend('cpu')):
            state=restore_state(root/receipt['checkpoint'],p);validate_bias_moments(state)
            parts=lr._parts(root,p);model=lr.EnergyModel(lr.ModelConfig(**p['model']))
            a=predictions(state,parts,model,c['microbatch_size'],j,store,'child')
            with np.load(root/receipt['predictions'],allow_pickle=False) as expected:
                maximum=max(float(np.max(np.abs(a[k]-expected[k]))) for k in a)
            if maximum>1e-10:raise ValueError('Child endpoint prediction differs')
        j.check()
        if cc.canonical_hash(c)==CONFIG_ID:verify_inputs(c);verify_gate(c,gate)
        else:cc.verify_pins(source['pins'])
        owner.check()
        if attempt['binding']!=probe_binding(root):raise ValueError('Child inputs changed during probe')
        record={'status':'passed','binding':attempt['binding'],'state_hashes':lr.state_hashes(state),
            'full_hash':full_hash(state),'adam_count':int(state['optimizer'][0].count),'prediction_frames':sum(len(p.record_ids) for p in parts.values()),
            'maximum_absolute_error':maximum,'random_initializations':0,'training_updates':0,'work':j.work}
        owner.json('checkpoint-verification.json',record,exclusive=True)
        owner.json('probe-completion.json',{'status':'passed','attempt_sha256':entry['attempt_sha256'],
            'verification_sha256':cc.digest(root/'checkpoint-verification.json'),'ended_at_utc':cc.utc()},exclusive=True)
        j.check();return record
    except BaseException as exc:
        if owner.owns():owner.json('probe-failure.json',{'status':'failed','reason':type(exc).__name__+': '+str(exc),'ended_at_utc':cc.utc()})
        raise
    finally:store.close();owner.close()

def run_owned(c,source,gate,code,owner,start,*,probe_runner=launch_probe):
    """Private engine accepts tiny engineering support; public config never bypasses guards."""
    store=Store(owner);j=Journal(owner,c['total_wall_seconds'],c['finalization_reserve_seconds'],start=start)
    history=[];state=None;prov=None;receipt=None;accepted=0;calibrations=0;oracles={};pending_update=False
    out={'status':'failed','execution_valid':False,'random_initializations':0,'reference':'historical_width32',
        'started_at_utc':cc.utc(),'original_baselines':source['provenance']['original_baselines']}
    try:
        j.check()
        import jax
        import numpy as np
        from flax import serialization
        from force_consistency import late_rate as lr
        from force_consistency.runs import save_checkpoint
        import offset_profile_core as core
        root=owner.path;original=source['provenance'];norm=lr.Normalization(**original['normalization'])
        parts=lr._parts(source['root'],original)
        if any(p.positions.shape[1]!=c['atom_count'] for p in parts.values()):raise ValueError('Nonconstant/unreviewed atom count')
        model=lr.EnergyModel(lr.ModelConfig(**original['model']))
        owner.json('config.json',c);owner.json('code.json',code);owner.json('prelaunch-gate.json',gate);owner.json('source-pins.json',source['pins'])
        owner.json('shared-zero-provenance.json',original)
        owner.write('shared-zero.msgpack',(Path(source['root'])/'snapshots/0.msgpack').read_bytes(),exclusive=True)
        zero=core.restore_state(root/'shared-zero.msgpack',original);core.validate_bias_moments(zero);j.check()
        if lr.state_hashes(zero)!=original['initial_state_hashes']:raise ValueError('Restored shared-zero identities differ')
        source_full=core.full_hash(zero)
        for role in parts:owner.write(role+'-inputs.npz',(Path(source['root'])/(role+'-inputs.npz')).read_bytes(),exclusive=True)
        prov={**original,'config_identity':cc.canonical_hash(c),'code':code,'source_root':source['root'],
            'source_zero_state_hashes':lr.state_hashes(zero),'source_zero_full_hash':source_full,
            'source_zero_payload_sha256':core._read_checkpoint(root/'shared-zero.msgpack',original)['sha256'],
            'source_provenance_sha256':cc.digest(root/'shared-zero-provenance.json'),
            'source_blueprint':source['zero']['params_schema'],'gate_identity':cc.canonical_hash(gate)}
        bindings=projection_binding(prov,gate,'initial',0,0)
        context={'phase':'initial','incoming':0,'outgoing':0,'incoming_full_hash':source_full,'projection_name':None}
        def retain(event):
            # Latest raw work slots are overwritten after success, and retained on failure.
            # Every accepted projection vector/receipt and scored state is immutable.
            data=event.get('data');name=context['phase']+'-'+event['event']
            if data is not None:
                store.write('staging',name+'.msgpack',serialization.to_bytes(data))
            if event['event']=='projection':
                basename=context['phase']+'-'+str(context['outgoing'])
                sha=store.npz('projections',basename+'.npz',data,True)
                record={**projection_binding(prov,gate,context['phase'],context['incoming'],context['outgoing']),
                    **event['receipt'],'incoming_full_hash':context['incoming_full_hash'],'vector_sha256':sha}
                store.json('projections',basename+'.json',record,True);context['projection_name']='projections/'+basename+'.json'
            elif 'receipt' in event:store.json('staging',name+'.json',event['receipt'])
        def score(label,publish):
            nonlocal receipt
            j.phase=label;j.update=int(state['update'])
            arrays=predictions(state,parts,model,c['microbatch_size'],j,store,label);m=metrics(parts,arrays)
            if not publish:return arrays,m
            u=int(state['update']);core.validate_bias_moments(state);state_fields_equal(state,zero)
            row={'update':u,'metrics':m,'state_hashes':lr.state_hashes(state),'full_hash':core.full_hash(state),
                'projection':context['projection_name'],'projection_sha256':cc.digest(root/context['projection_name']),
                'accepted_receipt':None if u==0 else 'accepted/'+str(u)+'.json'}
            for role in parts:row[role+'_gate']=lr.gate_result(m[role],original['original_baselines'][role],c['gate_fraction'])
            if u:row['accepted_receipt_sha256']=cc.digest(root/row['accepted_receipt'])
            history.append(row)
            checkpoint=store.child('snapshots');checkpoint_name=str(u)+'.msgpack'
            if (checkpoint.path/checkpoint_name).exists():raise FileExistsError('Immutable score checkpoint exists')
            with checkpoint.directory():save_checkpoint(Path(checkpoint_name),state,prov)
            store.json('histories',str(u)+'.json',history,True)
            predsha=store.npz('predictions',str(u)+'.npz',arrays,True)
            receipt={'update':u,'checkpoint':'snapshots/'+checkpoint_name,'checkpoint_sha256':cc.digest(checkpoint.path/checkpoint_name),
                'history':'histories/'+str(u)+'.json','history_sha256':cc.digest(root/'histories'/ (str(u)+'.json')),
                'predictions':'predictions/'+str(u)+'.npz','predictions_sha256':predsha,
                'state_hashes':row['state_hashes'],'full_hash':row['full_hash'],'row':row}
            store.json('commits',str(u)+'.json',receipt,True);owner.json('commit.json',receipt)
            j.save();j.check();print(json.dumps({'update':u,'energy_fraction':row['fitting_gate']['fraction_remaining']['energy_mse_eV2'],
                'force_fraction':row['fitting_gate']['fraction_remaining']['force_mse_eV2_per_A2'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
        state=zero;j.phase='shared_zero_replay'
        replay,raw=score('shared_zero_replay',False)
        store.npz('replay','shared-zero.npz',replay,True);owner.json('shared-zero-replay.json',raw)
        for role in parts:
            for k in cc.MSE_KEYS:
                if not np.isclose(raw[role][k],original['original_baselines'][role][k],atol=1e-12,rtol=1e-10):raise ValueError('Original zero baseline replay differs')
        batches=[lr.partition_batch(parts['fitting'],np.arange(i,min(i+c['microbatch_size'],len(parts['fitting'].record_ids))))
            for i in range(0,len(parts['fitting'].record_ids),c['microbatch_size'])]
        full=lr.partition_batch(parts['fitting'],np.arange(len(parts['fitting'].record_ids)))
        j.phase='initial';pp,initial_projection=core.calibrate(zero['params'],batches,model,j.call,retain)
        state={**zero,'params':pp};calibrations=1;prov['initial_state_hashes']=lr.state_hashes(state);prov['initial_full_hash']=core.full_hash(state)
        owner.json('provenance.json',prov)
        owner.json('initial-projection-state.json',{'projection':context['projection_name'],'source_full_hash':source_full,
            'projected_full_hash':core.full_hash(state),'projected_state_hashes':lr.state_hashes(state)})
        score('score',True);j.phase='compile';core.prepare(state,batches,model,norm,c['loss_weights'],c['learning_rate'],j.check)
        j.phase='warmup';context.update(phase='warmup',incoming=0,outgoing=1,incoming_full_hash=core.full_hash(state),projection_name=None)
        before=core.full_hash(state);warm,g,wr=core.profile_step(state,batches,model,norm,c['loss_weights'],c['learning_rate'],j.call,retain)
        if core.full_hash(state)!=before:raise ValueError('Warmup mutated projected zero')
        owner.json('warmup.json',{'status':'discarded','receipt':wr,'projection':context['projection_name'],'incoming_full_hash':before})
        for u in range(1,c['max_updates']+1):
            j.check();j.phase='training';j.update=u;pending_update=True
            context.update(phase='training',incoming=u-1,outgoing=u,incoming_full_hash=core.full_hash(state),projection_name=None)
            candidate,g,r=core.profile_step(state,batches,model,norm,c['loss_weights'],c['learning_rate'],j.call,retain)
            state_fields_equal(candidate,zero)
            if u in c['oracle_updates']:
                j.phase='oracle';j.check()
                if u>1 and not any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(state['optimizer'][0].mu)):raise ValueError('Oracle requires nonzero incoming body moments')
                def oracle_retain(e):
                    if 'data' in e:store.write('oracles',str(u)+'-'+e['event']+'.msgpack',serialization.to_bytes(e['data']),True)
                    if 'receipt' in e:store.json('oracles',str(u)+'-'+e['event']+'.json',e['receipt'],True)
                    if e['event']=='oracle_reference_projection':
                        sha=store.npz('oracles',str(u)+'-projection.npz',e['data'],True)
                        record={**projection_binding(prov,gate,'oracle-reference',u-1,u),**e['receipt'],
                            'incoming_full_hash':core.full_hash(state),'actual_candidate_full_hash':core.full_hash(candidate),'vector_sha256':sha}
                        store.json('oracles',str(u)+'-projection.json',record,True)
                oracle=core.profile_oracle(state,candidate,g,full,model,norm,c['loss_weights'],c['learning_rate'],j.call,oracle_retain)
                oracle['projection_binding']=projection_binding(prov,gate,'oracle-reference',u-1,u)
                oracle['artifact_sha256']={n.name:cc.digest(n) for n in store.child('oracles').path.glob(str(u)+'-*')}
                store.json('oracles',str(u)+'.json',oracle,True);oracles[str(u)]=oracle
            j.check();r.update(projection=context['projection_name'],projection_sha256=cc.digest(root/context['projection_name']),
                config_identity=prov['config_identity'],gate_identity=prov['gate_identity'],ordinal=1)
            store.json('accepted',str(u)+'.json',r,True)
            state=candidate;accepted+=1;calibrations+=1;pending_update=False;j.accepted=accepted;j.phase='training';j.save()
            if u in c['score_updates']:score('score',True)
        j.check();out.update(endpoint=history[-1],history=history)
        if c['max_updates']==6000:
            j.phase='child';attempt=reserve_probe(root,j.deadline-j.reserve)
            child=probe_runner(root,j.deadline-j.reserve-time.perf_counter(),attempt['token'])
            if child['status']!='passed' or child['binding']!=probe_binding(root):raise ValueError('Child binding changed')
            out['checkpoint_verification']=child
        else:
            # Private tiny integration fixture only; public canonical guard fixes6000.
            out['checkpoint_verification']={'status':'private_engineering_fixture_no_production_probe'}
        j.check(final=True)
        if c['max_updates']==6000:verify_inputs(c);verify_gate(c,gate);cc.verify_code(code)
        if core.full_hash(zero)!=source_full:raise ValueError('Shared zero mutated')
        out.update(status='completed',execution_valid=True,stop_reason='fixed_budget')
    except BaseException as exc:
        out.update(status='incomplete' if isinstance(exc,TimeoutError) else 'failed',execution_valid=False,
            stop_reason='wall_budget' if isinstance(exc,TimeoutError) else 'failure',failure_reason=type(exc).__name__+': '+str(exc))
        if state is not None and owner.owns():
            from flax import serialization
            owner.write('failure-last-accepted.msgpack',serialization.to_bytes(state))
    finally:
        out.update(updates=accepted,accepted_calibrations=calibrations,pending_update_not_accepted=pending_update,
            persisted_update=receipt['update'] if receipt else None,oracles=oracles,work=j.work,
            pending_inflight_unknown=j.pending,late_call=j.late_call,elapsed_seconds=time.perf_counter()-start,
            wall_seconds=c['total_wall_seconds'],ended_at_utc=cc.utc())
        if out['elapsed_seconds']>=c['total_wall_seconds']:out.update(status='incomplete',execution_valid=False,stop_reason='wall_budget')
        if owner.owns():finalize_summary(j,out)
        else:out.update(status='failed',execution_valid=False,failure_reason='Root directory ownership lost')
        store.close()
    return out

def execute(c):
    # Start charged clock immediately after exclusive reservation, before pure guards/imports.
    if cc.canonical_hash(c)!=CONFIG_ID:raise ValueError('Unreviewed public configuration')
    owner=reserve_root(c['run_directory']);start=time.perf_counter()
    try:
        owner.json('budget-start.json',{'started_at_utc':cc.utc(),'wall_seconds':3600,'reserve_seconds':180})
        source=verify_inputs(c);gate=verify_gate(c);verify_invocation(c,gate)
        from force_consistency.runs import code_provenance
        code=code_provenance(Path.cwd());cc.verify_code(code)
        validate_launch_code(code,{'code':code},gate)
        import jax
        from force_consistency.runs import require_backend
        with jax.default_device(require_backend('cpu')):return run_owned(c,source,gate,code,owner,start)
    except BaseException as exc:
        if owner.owns() and not (owner.path/'summary.json').exists():owner.json('summary.json',{'status':'failed','execution_valid':False,
            'failure_reason':type(exc).__name__+': '+str(exc),'updates':0,'random_initializations':0,'elapsed_seconds':time.perf_counter()-start})
        raise
    finally:owner.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/aspirin-offset-profile.json');p.add_argument('--probe');p.add_argument('--token');a=p.parse_args()
    if a.probe:probe(a.probe,a.token);return 0
    result=execute(cc.read(a.config));return 0 if result['execution_valid'] else 1
if __name__=='__main__':raise SystemExit(main())
