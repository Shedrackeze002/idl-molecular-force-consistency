"""Shared fixed settings and small provenance utilities for the E3x phase."""
from pathlib import Path
import hashlib
import json
import math
import os
import platform
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
SPLIT='73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a'
BALANCED={'energy':0.014584365670390719,'force':1.9854156343296092}
RECIPES={'A':(32,.001,{'energy':1.,'force':1.}), 'B':(64,.003,{'energy':1.,'force':1.}), 'C':(64,.003,BALANCED)}
ARMS=('energy','direct','consistency')
HELPERS=('e3x_phase_common.py','benchmark_e3x_budget.py','estimate_e3x_budget.py','evaluate_e3x_adequacy.py','launch_e3x_candidates.py')


def digest(data):return hashlib.sha256(data).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def save_json(path,value):Path(path).write_bytes(encoded(value))
def fresh_directory(path):Path(path).mkdir(parents=True,exist_ok=False);return Path(path)


def require_equal_pin(actual,expected):
    if actual!=expected:raise ValueError('Source/dependency/helper pin mismatch')


def source_pin():
    import subprocess
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
    files={n:digest((ROOT/n).read_bytes()) for n in sorted(set(names)) if n and (n.startswith('src/') or n in ('pyproject.toml','uv.lock')) and (ROOT/n).is_file()}
    return {'code_fingerprint':digest(json.dumps(files,sort_keys=True,separators=(',',':'),allow_nan=False).encode()),
            'code_files':files,'lock_sha256':files['uv.lock'],'python':platform.python_version(),
            'dependencies':{n:version(n) for n in ('jax','jaxlib','flax','e3x','optax','numpy','ase')},
            'runtime_cpu':{'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'thread_environment':{n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','XLA_FLAGS')}},
            'helpers':{n:digest((ROOT/'scripts'/n).read_bytes()) for n in HELPERS},
            'templates':{n.name:digest(n.read_bytes()) for n in sorted((ROOT/'configs').glob('aspirin-e3x-*.json'))}}


def child_environment():
    """Pin children to this checkout even if the calling shell omitted PYTHONPATH."""
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join((str(ROOT/'src'),str(ROOT/'scripts')))
    return env


def assert_imports():
    import force_consistency.train as train
    if Path(train.__file__).resolve()!=ROOT/'src/force_consistency/train.py':raise ValueError('Wrong imported trainer')


def fixed_config(recipe, data, output, updates, wall_seconds, *,arm='energy',strength=0.,seed=0):
    width,lr,weights=RECIPES[recipe]
    if type(updates) is not int or updates<1 or not math.isfinite(wall_seconds) or wall_seconds<=0:raise ValueError('Invalid budget')
    if arm not in ARMS or seed not in (0,1):raise ValueError('Unknown arm/seed')
    if (arm=='consistency' and strength not in (.1,1.,10.)) or (arm!='consistency' and strength!=0):raise ValueError('Invalid strength')
    return {'data_directory':str(Path(data).resolve()),'expected_split_hash':SPLIT,'run_directory':str(Path(output).resolve()),
            'resume_checkpoint':None,'expected_learning_subset_hash':None,'mode':'integration','backend':'cpu','seed':seed,'batch_size':4,'learning_rate':lr,'validation_interval':500,
            'max_updates':updates,'wall_seconds':wall_seconds,'arm':arm,'consistency_strength':strength,'loss_weights':dict(weights),
            'model':{'features':width,'max_degree':1,'iterations':2,'radial_basis':8,'cutoff_A':5.,'energy_readout_init':'zero'}}


def dataset_pin(directory):
    from force_consistency.data.prepare import _validated_manifest
    m=_validated_manifest(directory,expected_split_hash=SPLIT)
    return {'manifest_identity':m['identity'],'files':{n:digest((Path(directory)/n).read_bytes()) for n in ('manifest.json','train.npz','validation.npz','validation_minima_diagnostics.npz')}}


def label_coverage(report,manifest):
    groups=manifest['splits']['groups']['aspirin'];connections=report['connections']
    if report['manifest_identity']!=manifest['identity']:raise ValueError('Label manifest mismatch')
    for part in ('train','validation'):
        if {n for n,v in connections.items() if v['split']==part}!=set(groups[part]):raise ValueError('Label coverage mismatch')
    if len(connections)!=31 or sorted(report['test_connections_skipped'])!=sorted(groups['test']):raise ValueError('Label partition mismatch')
    if any(set(v['branches'])!={'plus','minus'} or any(b['pairs']<=0 for b in v['branches'].values()) for v in connections.values()):raise ValueError('Missing label branches')
    return True
