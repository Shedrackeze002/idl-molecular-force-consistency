import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
torch.set_num_threads(4)
from geom import *
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from mace.calculators import MACECalculator
MODEL='Group9/deliverables/idris-mace-2026-10-18/mace-idris-handoff/model/final_epoch.model'
calc=MACECalculator(model_paths=MODEL, device='cpu', default_dtype='float64')
sp=species(); mins=minima(); T=json.load(open(os.path.join(os.path.dirname(__file__),'..','targets.json')))
class Counting:
    def __init__(s): s.n=0
cnt=Counting()
_orig=calc.calculate
def counted(*a,**k):
    cnt.n+=1; return _orig(*a,**k)
calc.calculate=counted
def evaluate(a,t):
    f=a.get_forces(); X=a.get_positions()
    r,_=sym_rmsd(sp,X,mins[str(t)])
    best=(99,None)
    for k,m in mins.items():
        rr,_=sym_rmsd(sp,X,m)
        if rr is not None and rr<best[0]: best=(rr,k)
    return dict(max_force_component=float(np.abs(f).max()),rmsd_to_target=r,nearest_min=best[1],nearest_rmsd=best[0])
opts={'LBFGS':LBFGS,'BFGS':BFGS}
for oname in sys.argv[1:]:
    for sid,t in T.items():
        _,c,br=sid.split('/'); s,X,F,e=read_xyz(branch_frames(c,br)[0])
        a=Atoms(sp,positions=X.copy()); a.calc=calc; cnt.n=0; t0=time.time()
        opt=opts[oname](a, logfile=None)
        for fmax in (0.01, 0.001):
            conv=opt.run(fmax=fmax, steps=1500)
            ev=evaluate(a,t)
            rec=dict(optimizer=oname,fmax_norm=fmax,start=sid,target=t,converged=bool(conv),steps=int(opt.nsteps),force_calls=cnt.n,**ev,
                     recovered=bool(conv and ev['max_force_component']<=0.01 and ev['rmsd_to_target'] is not None and ev['rmsd_to_target']<=0.05),seconds=round(time.time()-t0,1))
            print(json.dumps(rec),flush=True)
print('DONE',flush=True)
