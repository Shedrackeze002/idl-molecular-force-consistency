import numpy as np, networkx as nx, glob, re, json, os
from ase.data import atomic_numbers, covalent_radii
from networkx.algorithms.isomorphism import GraphMatcher
BASE=os.path.join(os.path.dirname(__file__),'..','data','Landscape17','aspirin')
def read_xyz(f):
    L=open(f).read().split('\n'); n=int(L[0])
    sp=[];X=[];F=[]
    for l in L[2:2+n]:
        p=l.split(); sp.append(p[0]); X.append([float(v) for v in p[1:4]])
        F.append([float(v) for v in p[4:7]] if len(p)>=7 else [np.nan]*3)
    e=float(re.findall(r'energy=(\S+)',L[1])[0]) if 'energy=' in L[1] else None
    return sp,np.array(X),np.array(F),e
def species():
    sp,_,_,_=read_xyz(os.path.join(BASE,'aspirin.xyz')); return sp
def minima():
    rows=[np.array([float(v) for v in l.split()]).reshape(-1,3) for l in open(os.path.join(BASE,'min.coords')) if l.strip()]
    return {str(i):r for i,r in enumerate(rows)}
def branch_frames(conn,br):
    fs=sorted(glob.glob(os.path.join(BASE,'SteepestDescent',conn,f'coords{br}_*.xyz')),key=lambda s:int(re.findall(r'_(\d+)\.xyz',s)[0]))
    return fs
def graph(sp,X,factor=1.2):
    r=np.array([covalent_radii[atomic_numbers[z]] for z in sp]); d=np.linalg.norm(X[:,None]-X[None],axis=-1)
    g=nx.Graph(); g.add_nodes_from((i,{'s':z}) for i,z in enumerate(sp))
    g.add_edges_from((i,j) for i,j in zip(*np.where(np.triu((d>0.1)&(d<=factor*(r[:,None]+r[None])),1))))
    return g
def kabsch_rmsd(P,Q,allow_reflection=True):
    P=P-P.mean(0); Q=Q-Q.mean(0); H=P.T@Q; U,S,Vt=np.linalg.svd(H)
    if allow_reflection: R=(U@Vt).T
    else:
        dd=np.sign(np.linalg.det(Vt.T@U.T)); D=np.diag([1,1,dd]); R=Vt.T@D@U.T
    return np.sqrt(((P@R.T-Q)**2).sum(1).mean())
def sym_rmsd(sp,X,Ref):
    """min RMSD over species-preserving graph isomorphisms of endpoint->reference, O(3) alignment.
    returns (rmsd, n_isomorphisms); rmsd=None if graphs not isomorphic."""
    gX=graph(sp,X); gR=graph(sp,Ref)
    gm=GraphMatcher(gR,gX,node_match=lambda a,b:a['s']==b['s'])
    best=None;n=0
    for m in gm.isomorphisms_iter():   # m: ref idx -> X idx
        n+=1; perm=[m[i] for i in range(len(sp))]
        r=kabsch_rmsd(X[perm],Ref); best=r if best is None or r<best else best
        if n>2000: break
    return best,n
def plain_rmsd_all(sp,X,mins):
    return {k:kabsch_rmsd(X,v) for k,v in mins.items()}
