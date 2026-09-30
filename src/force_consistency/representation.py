"""Necessary rigid-symmetry constraints; these probes never replace force labels."""
import numpy as np
from .coverage import numeric_path


def _positions(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('positions must be finite, nonempty (N,3)')
    return a


def rigid_projection(positions,forces,rcond=1e-12):
    p=_positions(positions);f=np.asarray(forces,dtype=np.float64)
    if f.shape!=p.shape or not np.isfinite(f).all() or not 0<rcond<1:
        raise ValueError('invalid forces or SVD rank tolerance')
    r=p-p.mean(axis=0)
    q=np.column_stack([np.tile(e,(len(p),1)).ravel() for e in np.eye(3)]+
                      [np.cross(e,r).ravel() for e in np.eye(3)])
    norms=np.linalg.norm(q,axis=0);qn=q[:,norms>0]/norms[norms>0]
    u,s,_=np.linalg.svd(qn,full_matrices=False);rank=int(np.count_nonzero(s>rcond*s[0]))
    basis=u[:,:rank];parallel=(basis@(basis.T@f.ravel())).reshape(f.shape)
    compatible=f-parallel
    raw=float(np.mean(f*f));floor=float(np.mean(parallel*parallel));cm=float(np.mean(compatible*compatible))
    return {'parallel':parallel,'compatible':compatible,'rank':rank,'singular_values':s.tolist(),
            'net_force':f.sum(0).tolist(),'centered_torque':np.cross(r,f).sum(0).tolist(),
            'normalized_column_dots':(qn.T@f.ravel()).tolist(),
            'raw_mse':raw,'translation_mse':float(np.sum(f.sum(0)**2)/(3*len(f)**2)),
            'floor_mse':floor,'compatible_mse':cm,
            'checks':{'reconstruction_max':float(np.max(abs(parallel+compatible-f))),
                      'orthogonality_max':float(np.max(abs(basis.T@compatible.ravel()))),
                      'pythagorean_error':raw-floor-cm}}


def cutoff_graph(positions,cutoff_A=5.,hops=2):
    p=_positions(positions)
    if not np.isfinite(cutoff_A) or cutoff_A<=0 or not isinstance(hops,int) or hops<1:
        raise ValueError('invalid graph cutoff or hops')
    n=len(p);dist=np.linalg.norm(p[:,None]-p[None,:],axis=-1);nonself=~np.eye(n,dtype=bool)
    edges=(dist<cutoff_A)&nonself;degree=edges.sum(1)
    reach=edges.copy();front=edges.copy()
    for _ in range(1,hops):
        front=(front.astype(np.int64)@edges.astype(np.int64))>0;reach|=front
    reachable=int(np.count_nonzero(reach&nonself));denominator=n*(n-1)
    unseen=set(range(n));components=0
    while unseen:
        stack=[unseen.pop()];components+=1
        while stack:
            neighbors=set(np.flatnonzero(edges[stack.pop()]))&unseen
            unseen-=neighbors;stack.extend(neighbors)
    return {'atom_count':n,'cutoff_A':float(cutoff_A),'hops':hops,'degree_min':int(degree.min()),
            'degree_max':int(degree.max()),'isolated_atoms':int(np.count_nonzero(degree==0)),
            'connected_components':components,'directed_edges':int(edges.sum()),
            'directed_pair_count':denominator,'directed_pair_density':float(edges.sum()/denominator) if denominator else None,
            'reachable_pairs':reachable,'two_hop_fraction':reachable/denominator if denominator else None,
            'pairs_near_cutoff':int(np.count_nonzero((abs(dist-cutoff_A)<=1e-8)&nonself))}


def partition_constraints(partition,predicted_forces=None):
    positions=np.asarray(partition.positions);forces=np.asarray(partition.forces)
    if positions.ndim!=3 or forces.shape!=positions.shape or len(positions)!=len(partition.record_ids) or not len(positions):
        raise ValueError('invalid partition shape/count')
    predicted=None if predicted_forces is None else np.asarray(predicted_forces)
    if predicted is not None and (predicted.shape!=forces.shape or not np.isfinite(predicted).all()):
        raise ValueError('invalid prediction shape/values')
    records=[];prediction=[]
    for i,rid in enumerate(partition.record_ids):
        d=rigid_projection(positions[i],forces[i]);_,_,branch,frame=numeric_path(rid)
        row={k:v for k,v in d.items() if k not in ('parallel','compatible')}
        row.update(record_id=rid,groups=list(partition.memberships[rid]['groups']),
                   role=partition.memberships[rid]['role'],branch=branch,frame=frame,
                   force_component_count=int(forces[i].size),graph=cutoff_graph(positions[i]))
        if predicted is not None:
            pd=rigid_projection(positions[i],predicted[i]);mse=float(np.mean((forces[i]-predicted[i])**2))
            residual=float(np.mean((d['compatible']-predicted[i])**2));error=mse-d['floor_mse']-residual
            prediction.append({'record_id':rid,'raw_mse':mse,'compatible_residual_mse':residual,
                'decomposition_error':error,'max_projected_component':float(np.max(abs(pd['parallel']))),
                'normalized_column_dots':pd['normalized_column_dots'],
                'valid':bool(np.max(abs(pd['parallel']))<=1e-8 and np.isclose(mse,d['floor_mse']+residual,atol=1e-10,rtol=1e-8))})
        records.append(row)
    keys=('raw_mse','translation_mse','floor_mse','compatible_mse')
    def pool(rows):
        weights=np.array([r['force_component_count'] for r in rows]);out={k:float(np.average([r[k] for r in rows],weights=weights)) for k in keys}
        out['floor_fraction_of_zero_force']=out['floor_mse']/out['raw_mse'] if out['raw_mse'] else None
        return out
    groups=sorted({g for r in records for g in r['groups']});bygroup={g:pool([r for r in records if g in r['groups']]) for g in groups}
    result={'record_ids':list(partition.record_ids),'record_count':len(records),'force_component_count':int(forces.size),
            'records':records,'pooled':pool(records),'groups':bygroup,
            'equal_group':{k:float(np.mean([g[k] for g in bygroup.values()])) for k in keys}}
    if predicted is not None:
        result['prediction_checks']={'valid':all(r['valid'] for r in prediction),'records':prediction,
                                    'max_projected_component':max(r['max_projected_component'] for r in prediction)}
    return result
