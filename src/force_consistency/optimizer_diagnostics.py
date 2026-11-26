"""Observations of immutable optimizer snapshots, with explicit completed-work accounting."""
import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .gradient_diagnostics import _objectives, _batches, gradient_geometry
from .coverage_runs import _state_hashes, _predict, _group_score, gate_result, MSE_KEYS
from .coverage import take_subset
from .loss_balance import accumulated_weighted_loss_and_grad, build_weighted_step, validate_weights
from .runs import assert_finite


def tree_norm(tree):
    return sum(float(np.vdot(x, x)) for x in jax.tree.leaves(tree)) ** .5


def _dot(a, b):
    if jax.tree.structure(a) != jax.tree.structure(b):
        raise ValueError('Direction tree mismatch')
    return sum(float(np.vdot(x, y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def tree_identity(a, b, atol=1e-10, rtol=1e-8):
    if jax.tree.structure(a) != jax.tree.structure(b):
        return {'valid': False, 'max_absolute_difference': None}
    pairs=list(zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
    return {'valid': all(np.allclose(x,y,atol=atol,rtol=rtol) for x,y in pairs),
            'max_absolute_difference': max((float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in pairs),default=0.)}


def adam_directions(params, optstate, gradient, learning_rate):
    assert_finite((params,optstate,gradient))
    if any(np.any(np.asarray(x)<0) for x in jax.tree.leaves(optstate[0].nu)):
        raise ValueError('Negative Adam variance')
    tx=optax.adam(learning_rate)
    delta,nextopt=tx.update(gradient,optstate,params)
    count=int(optstate[0].count)+1
    mu=jax.tree.map(lambda m,g:.9*m+.1*g,optstate[0].mu,gradient)
    nu=jax.tree.map(lambda n,g:.999*n+.001*g*g,optstate[0].nu,gradient)
    manual=jax.tree.map(lambda m,n:-learning_rate*(m/(1-.9**count))/(jnp.sqrt(n/(1-.999**count))+1e-8),mu,nu)
    checks={'delta':tree_identity(delta,manual),'mu':tree_identity(nextopt[0].mu,mu),'nu':tree_identity(nextopt[0].nu,nu)}
    target=tree_norm(delta)
    directions={'stateful_adam':{'delta':delta,'norm':target,'status':'available' if target else 'zero_direction'}}
    reset,resetopt=tx.update(gradient,tx.init(params),params)
    for name,raw in [('reset_adam_matched_norm',reset),('steepest_matched_norm',jax.tree.map(lambda g:-g,gradient))]:
        size=tree_norm(raw)
        available=bool(target and size)
        d=jax.tree.map(lambda x:x*(target/size),raw) if available else None
        directions[name]={'delta':d,'norm':tree_norm(d) if available else None,'status':'available' if available else 'unavailable',
                          'reason':None if available else ('zero_target_norm' if not target else 'zero_reference_norm')}
    verification={'valid':bool(all(v['valid'] for v in checks.values()) and int(nextopt[0].count)==count and int(resetopt[0].count)==1),
                  'next_count':int(nextopt[0].count),'reset_count':int(resetopt[0].count),'manual_checks':checks}
    assert_finite((delta,nextopt))
    return directions,verification


def direction_geometry(ge,gf,direction,weights):
    a,b=validate_weights(weights)
    def block(e,f,d):
        g=jax.tree.map(lambda x,y:a*x+b*y,e,f)
        en,fn,gn,dn=map(tree_norm,(e,f,g,d));es,fs=_dot(e,d),_dot(f,d)
        return {'energy_norm':en,'force_norm':fn,'weighted_gradient_norm':gn,'direction_norm':dn,
                'energy_slope':es,'force_slope':fs,'weighted_slope':a*es+b*fs,
                'energy_cosine':es/(en*dn) if en and dn else None,'force_cosine':fs/(fn*dn) if fn and dn else None,
                'weighted_cosine':_dot(g,d)/(gn*dn) if gn and dn else None,
                'zero_energy_gradient':en==0,'zero_force_gradient':fn==0,'zero_weighted_gradient':gn==0,'zero_direction':dn==0}
    result={'whole':block(ge,gf,direction)}
    if 'Backbone_0' in ge:result['backbone']=block(ge['Backbone_0'],gf['Backbone_0'],direction['Backbone_0'])
    if 'energy_head' in ge:
        for name,key in [('head_kernel','kernel'),('head_bias','bias')]:
            result[name]=block(ge['energy_head']['0+'][key],gf['energy_head']['0+'][key],direction['energy_head']['0+'][key])
    return result


def directional_fd(params,direction,ge,gf,evaluate,steps=(1e-4,1e-5),atol=1e-8,rtol=1e-4,on_progress=lambda d:None):
    norm=tree_norm(direction)
    if not norm:return {'status':'not_applicable_zero_direction','valid':None,'objectives':{}}
    unit=jax.tree.map(lambda x:x/norm,direction)
    result={'status':'running','objectives':{name:{'autodiff':_dot(g,unit),'steps':[]} for name,g in [('energy',ge),('force',gf)]}}
    for h in steps:
        pp=jax.tree.map(lambda p,d:p+h*d,params,unit);pm=jax.tree.map(lambda p,d:p-h*d,params,unit)
        fd=(np.asarray(evaluate(pp))-np.asarray(evaluate(pm)))/(2*h)
        assert_finite(fd)
        for i,name in enumerate(('energy','force')):
            cell=result['objectives'][name];expected=cell['autodiff'];actual=float(fd[i])
            cell['steps'].append({'step':h,'finite_difference':actual,'absolute_error':abs(actual-expected),
                                 'valid':bool(np.isclose(actual,expected,atol=atol,rtol=rtol))})
            cell['valid']=any(s['valid'] for s in cell['steps'])
        on_progress(result)
    result.update(status='completed',valid=all(c['valid'] for c in result['objectives'].values()))
    return result


def analyze_optimizer_snapshot(state,fitting,anchor,model,norm,weights,config):
    params=state['params'];size=config.get('microbatch_size',16)
    check=config.get('_check_budget',lambda:None);save=config.get('_save_partial',lambda d:None)
    before=_state_hashes(state);a,b=validate_weights(weights)
    result={'status':'running','update':int(state['update']),'adam_count':int(state['optimizer'][0].count),
            'state_hashes_before':before,'weights':weights,'training_frame_exposures':0,
            'counts':{'fd_cells':0,'fd_steps':0,'probes':0},
            'compute':{'operations':[],'completed_frame_evaluations':{}},'roles':{},'directions':{},'probes':[]}
    def keep():save(copy.deepcopy(result))
    keep();check()
    def operation(category,role,frames,fn):
        check()
        item={'category':category,'role':role,'planned_frames':frames,'status':'started','completed_frames':None}
        result['compute']['operations'].append(item);keep()
        value=fn();jax.block_until_ready(value)
        item.update(status='completed',completed_frames=frames)
        counts=result['compute']['completed_frame_evaluations'];counts[category]=counts.get(category,0)+frames
        keep();assert_finite(value);check()
        return value
    values,gradfunctions=_objectives(model,norm)
    gradients={};parts={'fitting':fitting,'anchor':anchor}
    for role,part in parts.items():
        total=len(part.record_ids);batches=_batches(part,size);gs=[];losses=[]
        for i,fn in enumerate(gradfunctions):
            combined=None
            for batch in batches:
                v=operation('gradient_'+('energy','force')[i],role,len(batch['energy']),lambda fn=fn,batch=batch:fn(params,batch))
                w=jax.tree.map(lambda x:x*len(batch['energy'])/total,v)
                combined=w if combined is None else jax.tree.map(lambda x,y:x+y,combined,w)
            losses.append(float(combined[0]));gs.append(combined[1])
        gradients[role]=gs
        result['roles'][role]={'record_ids':list(part.record_ids),'losses':dict(zip(('energy','force'),losses,strict=True)),
                              'weighted_loss':a*losses[0]+b*losses[1],'gradient_geometry':gradient_geometry(*gs)}
    ge,gf=gradients['fitting'];g=jax.tree.map(lambda x,y:a*x+b*y,ge,gf)
    combined=None;total=len(fitting.record_ids);batches=_batches(fitting,size)
    for batch in batches:
        out=operation('weighted_gradient','fitting',len(batch['energy']),lambda batch=batch:accumulated_weighted_loss_and_grad(params,[batch],model,norm,weights))
        w=jax.tree.map(lambda x:x*len(batch['energy'])/total,out)
        combined=w if combined is None else jax.tree.map(lambda x,y:x+y,combined,w)
    result['weighted_gradient_identity']=tree_identity(g,combined[1])
    directions,verification=adam_directions(params,state['optimizer'],g,config['learning_rate'])
    result['adam_identity']=verification
    builder=build_weighted_step(model,norm,config['learning_rate'],weights)
    pp,oo,_=operation('reference_builder','fitting',total,lambda:builder(params,state['optimizer'],batches))
    delta=directions['stateful_adam']['delta'];proposed=jax.tree.map(lambda p,d:p+d,params,delta)
    _,expectedopt=optax.adam(config['learning_rate']).update(g,state['optimizer'],params)
    identities={'params':tree_identity(pp,proposed),'optimizer':tree_identity(oo,expectedopt)}
    result['builder_identity']={**identities,'valid':all(c['valid'] for c in identities.values())}
    if not all([verification['valid'],result['weighted_gradient_identity']['valid'],result['builder_identity']['valid']]):
        keep();raise ValueError('Adam/weighted gradient/builder identity failed')
    for name,entry in directions.items():
        result['directions'][name]={k:v for k,v in entry.items() if k!='delta'}
        result['directions'][name]['roles']={}
        if entry['delta'] is None:
            result['directions'][name]['finite_differences']={'status':'unavailable','valid':None};continue
        d=entry['delta']
        for role,part in parts.items():
            er,fr=gradients[role];rd={'geometry':direction_geometry(er,fr,d,weights)}
            result['directions'][name]['roles'][role]=rd
            def evaluate(p):
                out=np.zeros(2)
                for batch in _batches(part,size):
                    v=operation('fd',role,len(batch['energy']),lambda batch=batch:values(p,batch))
                    out+=np.asarray(v)*len(batch['energy'])/len(part.record_ids)
                return out
            def fd_progress(fd):
                rd['finite_differences']=copy.deepcopy(fd)
                result['counts']['fd_cells']=sum(len(v.get('finite_differences',{}).get('objectives',{})) for dd in result['directions'].values() for v in dd['roles'].values())
                result['counts']['fd_steps']=sum(len(c['steps']) for dd in result['directions'].values() for v in dd['roles'].values() for c in v.get('finite_differences',{}).get('objectives',{}).values())
                keep()
            fd=directional_fd(params,d,er,fr,evaluate,on_progress=fd_progress);rd['finite_differences']=fd;keep()
            if fd['valid'] is False:raise ValueError(f'Finite difference verification failed: {name}/{role}')
    def metrics(p,category):
        predictions={};scores={}
        for role,part in parts.items():
            es=[];fs=[]
            for batch in _batches(part,size):
                e,f=operation(category,role,len(batch['energy']),lambda batch=batch:_predict(model)(p,batch))
                es.append(np.asarray(e));fs.append(np.asarray(f).reshape(len(batch['energy']),*part.forces.shape[1:]))
            e=np.concatenate(es);f=np.concatenate(fs);predictions[role]=(e,f);scores[role]=_group_score(part,e,f)
        e,f=predictions['anchor'];overlap=set(fitting.record_ids)
        for label,fitted in [('anchor_fitted',True),('anchor_not_fitted',False)]:
            idx=[i for i,r in enumerate(anchor.record_ids) if (r in overlap)==fitted]
            scores[label]=_group_score(take_subset(anchor,[anchor.record_ids[i] for i in idx]),e[idx],f[idx])
        scores['anchor_fitted_flags']={r:r in overlap for r in anchor.record_ids}
        return scores
    result['source_metrics']=metrics(params,'source_predictions')
    for name,entry in directions.items():
        if entry['delta'] is None:continue
        for scale in (.1,1.):
            p=jax.tree.map(lambda p,d:p+scale*d,params,entry['delta']);m=metrics(p,'probe')
            point={'direction':name,'scale':scale,'metrics':m,'roles':{}}
            for role in parts:
                raw=m[role];baseline=config['_baselines'][role]
                point['roles'][role]={'original_baseline_fraction':gate_result(raw,baseline,.1)['fraction_remaining'],
                    'counterfactual_gate':gate_result(raw,baseline,.1),'weighted_loss':a*raw[MSE_KEYS[0]]/norm.energy_loss_scale_eV**2+b*raw[MSE_KEYS[1]]/norm.force_loss_scale_eV_per_A**2,
                    'source_relative_change':{key:raw[key]/result['source_metrics'][role][key]-1 if result['source_metrics'][role][key] else None for key in MSE_KEYS}}
            result['probes'].append(point);result['counts']['probes']+=1;keep()
    result['state_hashes_after']=_state_hashes(state)
    result['verification_valid']=before==result['state_hashes_after']
    result['status']='completed' if result['verification_valid'] else 'failed';keep();check()
    return result
