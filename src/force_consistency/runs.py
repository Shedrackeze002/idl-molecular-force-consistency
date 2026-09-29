"""Run provenance, finite-state gates and self-verifying atomic checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from importlib.metadata import version

from flax import serialization
import jax
import numpy as np

from .data.splits import canonical_hash


def require_backend(backend):
    if backend not in ('cpu','gpu'): raise ValueError('Backend must be cpu or gpu')
    try: devices=jax.devices(backend)
    except RuntimeError as exc: raise ValueError(f'Requested {backend.upper()} backend unavailable') from exc
    if not devices: raise ValueError(f'Requested {backend.upper()} backend unavailable')
    return devices[0]


def assert_finite(tree):
    for x in jax.tree.leaves(tree):
        if hasattr(x,'dtype') or isinstance(x,(int,float)):
            if not np.isfinite(np.asarray(x)).all(): raise FloatingPointError('Nonfinite update, state or metric')


def assert_parameter_dtype(params):
    if any(np.asarray(v).dtype!=np.float64 for v in jax.tree.leaves(params)): raise ValueError('Parameters must be float64')


def atomic_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        temp=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_checkpoint(path,state,provenance):
    assert_parameter_dtype(state['params']);assert_finite(state)
    payload=serialization.to_bytes(state)
    envelope={'payload':payload,'sha256':hashlib.sha256(payload).hexdigest(),'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))}
    atomic_write(path,serialization.msgpack_serialize(envelope))


def load_checkpoint(path,expected_provenance,*,template):
    try:
        envelope=serialization.msgpack_restore(Path(path).read_bytes())
        if envelope['provenance']!=json.dumps(expected_provenance,sort_keys=True,separators=(',',':')): raise ValueError('Checkpoint provenance mismatch')
        if hashlib.sha256(envelope['payload']).hexdigest()!=envelope['sha256']: raise ValueError('Checkpoint payload mismatch')
        state=serialization.from_bytes(template,envelope['payload'])
        assert_parameter_dtype(state['params']);assert_finite(state)
        return state
    except (KeyError,TypeError,UnicodeError) as exc: raise ValueError('Invalid checkpoint') from exc


def code_provenance(root):
    root=Path(root)
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root).decode().split('\0')
    files={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(set(names)) if n and (n.startswith('src/') or n in ('pyproject.toml','uv.lock')) and (root/n).is_file()}
    return {'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),'code_fingerprint':canonical_hash(files),'code_files':files,
            'lock_sha256':hashlib.sha256((root/'uv.lock').read_bytes()).hexdigest(),'python':platform.python_version(),
            'dependencies':{k:version(k) for k in ('jax','jaxlib','flax','e3x','optax','numpy','ase')}}
