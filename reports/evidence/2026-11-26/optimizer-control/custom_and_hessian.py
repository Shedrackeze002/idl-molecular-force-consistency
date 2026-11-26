import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, networkx as nx
torch.set_num_threads(4)
from geom import *
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
MODEL='Group9/deliverables/idris-mace-2026-10-18/mace-idris-handoff/model/final_epoch.model'
calc=MACECalculator(model_paths=MODEL, device='cpu', default_dtype='float64')
sp=species(); mins=minima(); T=json.load(open(os.path.join(os.path.dirname(__file__),'..','targets.json')))
def FE(X):
    a=Atoms(sp,positions=X); a.calc=calc
    return a.get_forces(), a.get_potential_energy()
def connected(X,cut=5.0):
    D=np.linalg.norm(X[:,None]-X[None],axis=-1); g=nx.Graph(); g.add_nodes_from(range(len(X)))
    g.add_edges_from((i,j) for i in range(len(X)) for j in range(i+1,len(X)) if D[i,j]<=cut)
    return nx.is_connected(g)
def nearest(X):
    best=(99,None)
    for k,m in mins.items():
        r,_=sym_rmsd(sp,X,m)
        if r is not None and r<best[0]: best=(r,k)
    return best
# Reimplementation of the report's original force-following procedure (methods.tex)
def custom(X, budget=1500):
    eta=0.2; n=0
    F,_=FE(X); n+=1
    while n<budget:
        if np.abs(F).max()<=0.01: return X,F,n,'converged'
        for trial in range(40):
            if n>=budget: break
            Xt=X+eta*F; Ft,_=FE(Xt); n+=1
            if np.abs(Ft).max()<=0.01 or float((Ft*F).sum())>=-1e-12:
                X,F=Xt,Ft; eta=min(2*eta,1.0); break
            eta=max(eta/2,1e-8)
    return X,F,n,('converged' if np.abs(F).max()<=0.01 else 'budget')
out={}
for sid in ['aspirin/13/plus','aspirin/66/plus']:
    _,c,br=sid.split('/'); s,X,F,e=read_xyz(branch_frames(c,br)[0])
    Xe,Fe,n,why=custom(X.copy())
    r,_=sym_rmsd(sp,Xe,mins[str(T[sid])]); nr=nearest(Xe)
    D=np.linalg.norm(Xe[:,None]-Xe[None],axis=-1)
    rec=dict(kind='custom',start=sid,requests=n,stop=why,maxF=float(np.abs(Fe).max()),connected_5A=connected(Xe),max_pair_dist=float(D.max()),rmsd_target=r,nearest=nr)
    print(json.dumps(rec),flush=True)
# Hessian check: 13/minus LBFGS endpoint vs MACE-relaxed min 3
def relax(X,fmax=1e-4):
    a=Atoms(sp,positions=X.copy()); a.calc=calc; LBFGS(a,logfile=None).run(fmax=fmax,steps=3000); return a.get_positions()
def hessian(X,h=1e-3):
    n=X.size; H=np.zeros((n,n)); x=X.ravel()
    for i in range(n):
        xp=x.copy(); xp[i]+=h; xm=x.copy(); xm[i]-=h
        fp,_=FE(xp.reshape(-1,3)); fm,_=FE(xm.reshape(-1,3))
        H[:,i]=-(fp.ravel()-fm.ravel())/(2*h)
    H=0.5*(H+H.T); w=np.linalg.eigvalsh(H); return w
_,c,br='aspirin/13/minus'.split('/'); s,X0,_,_=read_xyz(branch_frames(c,br)[0])
Xa=relax(X0); Xb=relax(mins['3'])
for name,Xq in [('13minus_endpoint',Xa),('min3_relaxed',Xb)]:
    w=hessian(Xq); F,E=FE(Xq)
    r3,_=sym_rmsd(sp,Xq,mins['3'])
    print(json.dumps(dict(kind='hessian',point=name,energy_eV=float(E),maxF=float(np.abs(F).max()),rmsd_to_min3=r3,lowest_eigs=[round(float(v),5) for v in w[:9]],n_negative=int((w<-1e-3).sum()))),flush=True)
ra,_=sym_rmsd(sp,Xa,Xb); print(json.dumps(dict(kind='pair',rmsd_endpoint_vs_relaxed_min3=ra)),flush=True)
