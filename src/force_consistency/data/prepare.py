"""Canonical physical arrays; the fitting loader has no test partition."""
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
from ase.data import atomic_numbers

from .audit import ConnectionInventory
from .source import SHA256
from .units import verify_units
from importlib.metadata import version
from .splits import PreparationError,apply_eligibility,assign_groups,canonical_hash


@dataclass(frozen=True)
class Normalization:
    energy_offset_eV: float
    energy_loss_scale_eV: float
    force_loss_scale_eV_per_A: float


@dataclass(frozen=True)
class PreparedPartition:
    positions: np.ndarray
    energy_residual: np.ndarray
    forces: np.ndarray
    atomic_numbers: np.ndarray
    record_ids: tuple
    memberships: dict


@dataclass(frozen=True)
class PreparedBundle:
    train: PreparedPartition
    validation: PreparedPartition
    validation_minima_diagnostics: PreparedPartition
    manifest: dict
    normalization: Normalization


def fit_normalization(train):
    energies=np.array([r.energy for r in train],dtype=np.float64)
    forces=np.stack([r.forces for r in train]).astype(np.float64)
    if not len(train) or not np.isfinite(energies).all() or not np.isfinite(forces).all(): raise PreparationError('Nonfinite training labels')
    return Normalization(float(energies.mean()),max(float(energies.std()),1e-3),max(float(np.sqrt(np.mean(forces**2))),1e-3))


def array_identity(a):
    dtype='<i8' if a.dtype.kind in 'iu' else '<f8'
    x=np.ascontiguousarray(a,dtype=dtype)
    return {'shape':list(x.shape),'dtype':dtype,'sha256':hashlib.sha256(x.tobytes()).hexdigest()}


def write_json(path,payload):
    Path(path).write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n')


def export_prepared(records,*,directory,inventory,endpoints,duplicates,policy,molecule,unit_evidence):
    verify_units(unit_evidence)
    rs=[r for r in records if r.molecule==molecule]
    if molecule not in inventory.connections: raise PreparationError('Unknown molecule')
    # Gate all selected endpoints before creating an output directory or split.
    for g in inventory.connections[molecule]:
        for b in ('plus','minus'):
            if endpoints.get(f'{molecule}/{g}/{b}',{}).get('status')!='matched': raise PreparationError(f'Unresolved endpoint {molecule}/{g}/{b}')
    inv=ConnectionInventory({molecule:inventory.connections[molecule]},{molecule:inventory.minima[molecule]}, {})
    splits=assign_groups(inv,policy);eligible=apply_eligibility(rs,splits,duplicates,endpoints)
    byid={r.record_id:r for r in rs};norm=fit_normalization([byid[i] for i in eligible.partitions['train']])
    natoms=len(rs[0].species);arrays={};metadata={}
    for part,ids in eligible.partitions.items():
        selected=[byid[i] for i in ids]
        if any(len(r.species)!=natoms or not np.isfinite(r.positions).all() or not np.isfinite(r.energy) or not np.isfinite(r.forces).all() for r in selected): raise PreparationError('Invalid physical arrays')
        a={'positions':np.array([r.positions for r in selected],dtype='<f8').reshape(-1,natoms,3),
           'forces':np.array([r.forces for r in selected],dtype='<f8').reshape(-1,natoms,3),
           'energy_residual':np.array([r.energy for r in selected],dtype='<f8')-norm.energy_offset_eV,
           'atomic_numbers':np.array([[atomic_numbers[s] for s in r.species] for r in selected],dtype='<i8').reshape(-1,natoms)}
        arrays[part]=a
        metadata[part]={'record_ids':ids,'arrays':{k:array_identity(v) for k,v in a.items()},'memberships':{i:eligible.memberships[i] for i in ids},'molecule':molecule,'source_sha256':{i:byid[i].source_sha256 for i in ids}}
    payload={'schema':1,'molecule':molecule,'archive_sha256':SHA256,'policy':asdict(policy),'policy_hash':canonical_hash(asdict(policy)),
             'split_hash':splits.identity,'splits':splits.payload(),'unit_evidence_hash':canonical_hash(unit_evidence),'unit_evidence':unit_evidence,
             'dependencies':{k:version(k) for k in ('numpy','scipy','ase','networkx')},
             'units':{'positions':'angstrom','energy':'eV','forces':'eV/angstrom'},'normalization':asdict(norm),
             'partitions':metadata,'excluded':eligible.excluded,'endpoints':{k:v for k,v in sorted(endpoints.items()) if k.startswith(molecule+'/')}}
    payload['identity']=canonical_hash(payload)
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()): raise PreparationError('Output directory is not empty')
    directory.mkdir(parents=True,exist_ok=True)
    # A success manifest is published only after all arrays have been written.
    for part,a in arrays.items():
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.npz',delete=False) as f:
            temp=Path(f.name);np.savez_compressed(f,**a)
        os.replace(temp,directory/f'{part}.npz')
    write_json(directory/'manifest.json',payload)
    return payload


def load_training_bundle(directory,*,expected_split_hash):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());identity=m.pop('identity')
    if canonical_hash(m)!=identity: raise PreparationError('Manifest identity mismatch')
    m['identity']=identity
    s=dict(m['splits']);sid=s.pop('identity')
    if canonical_hash(s)!=sid or sid!=expected_split_hash or m['split_hash']!=sid: raise PreparationError('Stale split hash')
    if m['archive_sha256']!=SHA256 or s['archive_sha256']!=SHA256: raise PreparationError('Stale archive hash')
    if canonical_hash(m['policy'])!=m['policy_hash'] or s['policy']!=m['policy']: raise PreparationError('Stale policy hash')
    if m['units']!={'positions':'angstrom','energy':'eV','forces':'eV/angstrom'}: raise PreparationError('Invalid units')
    if set(s['groups'])!={m['molecule']}: raise PreparationError('Mixed molecule manifest')
    verify_units(m['unit_evidence'])
    if canonical_hash(m['unit_evidence'])!=m['unit_evidence_hash']: raise PreparationError('Unit evidence identity mismatch')
    if any(v.get('status')!='matched' for v in m['endpoints'].values()): raise PreparationError('Unresolved preparation gate')
    partitions={}
    for part in ('train','validation','validation_minima_diagnostics'):
        info=m['partitions'][part]
        if info['molecule']!=m['molecule']: raise PreparationError('Mixed molecule arrays')
        with np.load(directory/f'{part}.npz',allow_pickle=False) as file:
            a={k:file[k] for k in info['arrays']}
        if any(array_identity(v)!=info['arrays'][k] or not np.isfinite(v).all() for k,v in a.items()): raise PreparationError('Array identity mismatch')
        n=len(info['record_ids'])
        if any(len(v)!=n for v in a.values()): raise PreparationError('Array count mismatch')
        for v in a.values(): v.setflags(write=False)
        partitions[part]=PreparedPartition(**a,record_ids=tuple(info['record_ids']),memberships=info['memberships'])
    return PreparedBundle(**partitions,manifest=m,normalization=Normalization(**m['normalization']))
