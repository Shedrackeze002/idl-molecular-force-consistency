import sys, os, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
torch.set_num_threads(1)
from geom import *
from ase import Atoms
from mace.calculators import MACECalculator
MODEL='Group9/deliverables/idris-mace-2026-10-18/mace-idris-handoff/model/final_epoch.model'
calc=MACECalculator(model_paths=MODEL, device='cpu', default_dtype='float64')
sp=species(); mins=minima(); T=json.load(open(os.path.join(os.path.dirname(__file__),'..','targets.json')))
for sid,t in T.items():
    _,c,br=sid.split('/'); fr=branch_frames(c,br); s,X,F,e=read_xyz(fr[0])
    a=Atoms(sp,positions=X.copy()); a.calc=calc
    f=a.get_forces()
    r,_=sym_rmsd(sp,X,mins[str(t)])
    print(json.dumps(dict(start=sid,nframes=len(fr),dft_maxcomp=float(np.abs(F).max()),dft_maxnorm=float(np.linalg.norm(F,axis=1).max()),mace_maxcomp=float(np.abs(f).max()),mace_maxnorm=float(np.linalg.norm(f,axis=1).max()),force_mae_meV=float(1000*np.abs(f-F).mean()),rmsd_start_to_target=r)),flush=True)
