from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import networkx as nx
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import cKDTree

from .types import FrameRecord, PreparationPolicy


@dataclass(frozen=True)
class AlignmentResult:
    rmsd: float
    permutation: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    search_complete: bool
    reason: str | None = None


@lru_cache(maxsize=20000)
def _graph(species, coordinates, factor, lower):
    p=np.frombuffer(coordinates,dtype=np.float64).reshape(-1,3)
    radii=np.array([covalent_radii[atomic_numbers[z]] for z in species])
    d=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    g=nx.Graph()
    g.add_nodes_from((i,{"species":z}) for i,z in enumerate(species))
    g.add_edges_from((i,j) for i,j in zip(*np.where(np.triu((d>lower)&(d<=factor*(radii[:,None]+radii[None,:])),1))))
    return nx.freeze(g)


def chemical_graph(record: FrameRecord, policy: PreparationPolicy) -> nx.Graph:
    return _graph(record.species,np.asarray(record.positions,dtype=np.float64).tobytes(),policy.graph_factor,policy.graph_min_distance)


def align_records(a: FrameRecord,b: FrameRecord,policy: PreparationPolicy) -> AlignmentResult:
    n=len(a.species)
    empty=AlignmentResult(float("inf"),np.arange(n),np.eye(3),np.zeros(3),False,"graph_mismatch")
    if Counter(a.species)!=Counter(b.species): return empty
    match=nx.algorithms.isomorphism.GraphMatcher(chemical_graph(a,policy),chemical_graph(b,policy),node_match=lambda x,y:x["species"]==y["species"])
    ac=a.positions-a.positions.mean(0);bc=b.positions-b.positions.mean(0)
    best=None
    for count,mapping in enumerate(match.isomorphisms_iter(),start=1):
        if count>policy.search_limit:
            return AlignmentResult(best.rmsd,best.permutation,best.rotation,best.translation,False,"search_limit") if best else empty
        order=np.array([mapping[i] for i in range(n)])
        u,_,vt=np.linalg.svd(ac.T@bc[order])
        for sign in (1.,-1.):
            d=np.eye(3);d[-1,-1]=sign;rotation=u@d@vt
            rmsd=float(np.sqrt(np.mean(np.sum((ac@rotation-bc[order])**2,axis=1))))
            if best is None or rmsd<best.rmsd or (rmsd==best.rmsd and tuple(order)<tuple(best.permutation)):
                best=AlignmentResult(rmsd,order,rotation,b.positions.mean(0)-a.positions.mean(0)@rotation,True)
    return best if best is not None else empty


def descriptor(record: FrameRecord) -> np.ndarray:
    blocks=defaultdict(list)
    for i in range(len(record.species)):
        for j in range(i+1,len(record.species)):
            key=tuple(sorted((record.species[i],record.species[j])))
            blocks[key].append(np.linalg.norm(record.positions[i]-record.positions[j]))
    return np.concatenate([np.sort(blocks[k]) for k in sorted(blocks)])


def candidate_pairs(records: Sequence[FrameRecord],epsilon: float) -> list[tuple[int,int]]:
    blocks=defaultdict(list)
    for i,r in enumerate(records): blocks[(r.molecule,tuple(sorted(Counter(r.species).items())))].append(i)
    out=[]
    for indices in blocks.values():
        if len(indices)<2: continue
        vectors=np.stack([descriptor(records[i]) for i in indices])
        radius=len(records[indices[0]].species)*epsilon+1e-10
        pairs=cKDTree(vectors).query_pairs(radius,p=2,eps=0)
        out.extend((indices[i],indices[j]) for i,j in sorted(pairs))
    return sorted(out)
