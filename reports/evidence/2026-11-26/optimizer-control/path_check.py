import sys,os,json,warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(__file__))
import numpy as np, torch
torch.set_num_threads(4)
from geom import *
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
MODEL='Group9/deliverables/idris-mace-2026-10-18/mace-idris-handoff/model/final_epoch.model'
calc=MACECalculator(model_paths=MODEL,device='cpu',default_dtype='float64')
sp=species(); mins=minima()
fr=branch_frames('13','minus'); s,X0,_,_=read_xyz(fr[0])
a=Atoms(sp,positions=X0.copy()); a.calc=calc; LBFGS(a,logfile=None).run(fmax=1e-4,steps=3000); Xe=a.get_positions()
best=(99,None,None)
for i,f in enumerate(fr):
    _,X,F,e=read_xyz(f); r,_=sym_rmsd(sp,Xe,X)
    if r<best[0]: best=(r,i,float(np.abs(F).max()))
print(json.dumps(dict(nframes=len(fr),closest_frame=best[1],rmsd=best[0],dft_maxF_at_frame=best[2])))
