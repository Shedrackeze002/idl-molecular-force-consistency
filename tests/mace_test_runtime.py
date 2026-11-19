"""Test-only extraction of the actual private runtime; never imported by production.

No runtime factory is exposed in the trust root. Tests compile its real function
ASTs into a disposable namespace and bind fakes there. This keeps tests on the
production code without installing mutable authorization hooks in production.
"""
import ast
from pathlib import Path
import types

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
