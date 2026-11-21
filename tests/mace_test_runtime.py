"""Test-only extraction of the actual private runtime; never imported by production.

No runtime factory is exposed in the trust root. Tests compile its real function
ASTs into a disposable namespace and bind fakes there. This keeps tests on the
production code without installing mutable authorization hooks in production.
"""
import ast
from pathlib import Path
import types


def inert_packet_sources(checkout, packet, attempt):
    """Build inert data; retain actual parity/loader and every admission check."""
    import hashlib
    import json
    import shutil
    import numpy as np
    from ase import Atoms
    from ase.io import write
    import mace_loader

    repo = SOURCE.parents[1]
    for name in ("scripts/check_mace_parity.py", "scripts/audit_mace_parity.py",
                 "scripts/mace_reference_adapter.py", "evaluation/adapter.py"):
        shutil.copyfile(repo / name, checkout / name)
    (packet / "data").mkdir(parents=True)
    (packet / "predictions").mkdir()
    (packet / "model").mkdir()
    model = packet / "model/final_epoch.model"
    model.write_bytes(b"INERT FIXTURE - NOT A SERIALIZED MODEL\n")
    groups = [group for group, count in {"13":165,"18":217,"18_2":227,"66":199,"78":166}.items()
              for _ in range(count)]
    ids = [f"inert-{i:04d}" for i in range(974)]
    frames = []
    for index in reversed(range(974)):
        positions = np.zeros((21,3),dtype=np.float64)
        positions[0,0] = index / 8
        atoms = Atoms("H21",positions=positions,pbc=False)
        atoms.info.update(record_id=ids[index], REF_energy=index/8)
        atoms.new_array("REF_forces",np.full((21,3),index/16,dtype=np.float64))
        frames.append(atoms)
    write(packet / "data/valid.extxyz",frames,format="extxyz",write_results=False)
    (packet / "data/valid.memberships.json").write_text(json.dumps(dict(zip(ids,groups))))
    order = np.roll(np.arange(974),137)
    np.savez(packet / "predictions/final-valid-predictions.npz",
        record_ids=np.array(ids)[order], energy=order.astype(np.float64)/8,
        forces=np.broadcast_to((order.astype(np.float64)/16)[:,None,None],(974,21,3)).copy())
    for index in range(7):
        (packet / f"data/inert-member-{index}.txt").write_text("inert packet member\n")
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    files=[{"path":str(path.relative_to(packet)),"bytes":path.stat().st_size,"sha256":sha(path)}
           for path in sorted(packet.rglob("*")) if path.is_file()]
    assert len(files)==11
    manifest=packet / "MANIFEST.json"
    manifest.write_text(json.dumps({"files":files}))
    loader=checkout / "scripts/mace_loader.py"
    lines=loader.read_text().splitlines()
    values={"MANIFEST_SHA256":sha(manifest),"MODEL_SHA256":sha(model),"MODEL_BYTES":model.stat().st_size}
    loader.write_text("\n".join(f"{key} = {values[key]!r}" if (key:=line.split(" = ")[0]) in values
                               else line for line in lines)+"\n")
    # The only learned-computation replacement is in this synthetic Git tree,
    # before its commit and policy are pinned. No production factory is added.
    body=f'''
check_liveness()
assert 'torch' not in sys.modules and 'mace' not in sys.modules
from ase.calculators.calculator import Calculator
import numpy as np
events=Path({str(attempt / 'inert-events.jsonl')!r})
def event(kind, index=None):
    with events.open('a') as stream:
        stream.write(json.dumps({{'kind':kind,'index':index,'pid':os.getpid()}})+'\\n')
event('construction')
assert snapshot.read_bytes()==b'INERT FIXTURE - NOT A SERIALIZED MODEL\\n'
class InertCalculator(Calculator):
    implemented_properties=['energy','forces']
    def get_forces(self,atoms=None):
        x=float(atoms.positions[0,0])
        event('forces',round(x*8))
        return np.full((len(atoms),3),x/2,dtype=np.float64)
    def get_potential_energy(self,atoms=None,force_consistent=False):
        x=float(atoms.positions[0,0])
        event('energy',round(x*8))
        return x
return InertCalculator()
'''
    runtime=checkout / "trust/mace_trusted_launcher.py"
    tree=ast.parse(runtime.read_text())
    construct=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=="construct")
    construct.body=ast.parse(body).body
    runtime.write_text(ast.unparse(tree)+"\n")
    return {"record_ids":np.array(ids)[order].tolist(),"indices":order.tolist(),
            "energy":order.astype(np.float64)/8,
            "forces":np.broadcast_to((order.astype(np.float64)/16)[:,None,None],(974,21,3)).copy()}


def inert_admission(checkout, commit):
    """Real platform/package/source inventories plus the synthetic source commit."""
    import hashlib
    import importlib.metadata
    import platform
    import subprocess
    import mace_loader
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def blob(revision,name):
        return subprocess.check_output(['git','-C',str(checkout),'rev-parse',revision+':'+name],text=True).strip()
    versions={dist.metadata['Name'].strip().lower().replace('_','-'):dist.version
              for dist in importlib.metadata.distributions()}
    sources=[]
    for name,(distribution,relative) in mace_loader.REQUIRED_SOURCES.items():
        path=Path(importlib.metadata.distribution(distribution).locate_file(relative))
        sources.append({'name':name,'distribution':distribution,'relative_path':relative,
                        'actual_sha256':sha(path),'original_sha256':sha(path),'matches_original':True})
    mace_dist=importlib.metadata.distribution('mace-torch')
    return {'status':'APPROVED','platform':{'python':platform.python_version(),'platform':platform.platform(),
            'architecture':platform.machine(),'glibc':platform.libc_ver()[1]},
        'installed_packages':[name+'=='+version for name,version in versions.items()],
        'installed_package_versions':versions,'authenticated_source_comparison':sources,
        'import_initializers':{name:sha(Path(mace_dist.locate_file(name))) for name in mace_loader.IMPORT_INITIALIZERS},
        'identities':{'implementation_commit':commit,
            'implementation_files':{name:{'bytes':(checkout/name).stat().st_size,'sha256':sha(checkout/name),
                                          'git_blob':blob(commit,name)} for name in mace_loader.REQUIRED_IMPLEMENTATION_FILES},
            'frozen_dependencies':{name:{'bytes':(checkout/name).stat().st_size,'sha256':sha(checkout/name),
                'baseline_commit':mace_loader.SOURCE_BASELINE,'git_blob':blob(mace_loader.SOURCE_BASELINE,name)}
                for name in mace_loader.FROZEN_DEPENDENCY_BLOBS}}}

SOURCE = Path(__file__).resolve().parents[1] / "trust/mace_trusted_launcher.py"


def _source():
    tree = ast.parse(SOURCE.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    return tree, main, imports


def supervisor_runtime():
    _, main, imports = _source()
    end = next(i for i, node in enumerate(main.body)
               if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    module = types.ModuleType("test_only_supervisor_runtime")
    module.__file__ = str(SOURCE)
    module.worker_spawn = None
    module.setup_state = {}
    tree = ast.Module(body=imports + main.body[:end], type_ignores=[])
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


def load_arm_with_fake_constructor(packet_root, attempt_root, *, constructor, expected_species=None):
    """Only this test file accepts an injected constructor."""
    import mace_loader
    tree, _, imports = _source()
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "load_arm")
    namespace = {"mace_loader": mace_loader, "construct": constructor,
                 "check_liveness": lambda *args: None}
    exec(compile(ast.Module(body=imports + [node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["load_arm"](packet_root, attempt_root, expected_species=expected_species)
