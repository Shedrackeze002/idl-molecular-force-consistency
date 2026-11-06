"""Plan v3 sections 4, 8, 10: what the launch modules never do, checked on their syntax tree rather than by grep
(a text search passes `**kwargs` and fails on a docstring that explains the rule)."""
import ast
from pathlib import Path

import pytest

from stage2_capped import admission

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH_MODULES = ("launch.py", "launch_a0.py", "launch_stage.py", "launch_arms.py", "launch_record.py", "launch_state.py",
                  "launch_checks.py", "replay_evidence.py", "verify_launch.py")
FORBIDDEN_KEYWORDS = {"allow_unledgered", "allow_custom_tasks", "allow_custom", "tasks"}
ALLOWED_NPZ = set(admission.DATASET_FILES) | {"best-validation.npz"}
LISTING_CALLS = {"glob", "rglob", "iterdir", "scandir", "listdir", "walk"}


def tree_of(name):
    return ast.parse((PACKAGE / name).read_text())


def calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == name) or (isinstance(n.func, ast.Name) and n.func.id == name))]


def test_the_stage_worker_calls_run_stage_with_exactly_five_keywords_and_none_of_the_test_flags():
    found = calls(tree_of("launch_stage.py"), "run_stage")
    assert len(found) == 1
    assert {k.arg for k in found[0].keywords} == {"ledger", "saved", "controller", "control_index", "saved_hashes"}
    assert all(k.arg is not None for k in found[0].keywords)                     # no **kwargs can hide a flag


def test_the_stage_worker_builds_the_control_index_with_stages_only():
    found = calls(tree_of("launch_stage.py"), "from_roots")
    assert len(found) == 1 and {k.arg for k in found[0].keywords} == {"stages"}


@pytest.mark.parametrize("name", LAUNCH_MODULES)
def test_no_launch_module_passes_a_test_flag_to_any_call(name):
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.Call):
            assert not (FORBIDDEN_KEYWORDS & {k.arg for k in node.keywords}), (name, node.lineno)


@pytest.mark.parametrize("name", LAUNCH_MODULES)
def test_no_launch_module_names_an_npz_other_than_the_five_known_files(name):
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".npz"):
            assert node.value in ALLOWED_NPZ or node.value == ".npz", (name, node.lineno, node.value)     # a bare suffix is not a file


@pytest.mark.parametrize("name", LAUNCH_MODULES)
def test_no_launch_module_lists_a_directory_through_a_data_path(name):
    """A heuristic: a listing call whose receiver or argument text mentions the data directory."""
    source = (PACKAGE / name).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            function = node.func
            attribute = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if attribute in LISTING_CALLS:
                text = ast.get_source_segment(source, node) or ""
                assert "data_directory" not in text and "data_dir" not in text, (name, node.lineno, text)


LISTING_ALLOWED = {("launch.py", "print_stage"), ("launch_state.py", "files"), ("launch_state.py", "entries"),
                   ("launch_state.py", "root_contents_problems"), ("launch_state.py", "marker_problems"),
                   ("launch_state.py", "violation_files"), ("launch_checks.py", "project_names"),
                   ("verify_launch.py", "load_records")}


def listing_calls(name):
    """(module, function) of every directory-listing call, found on the syntax tree whatever the variable is called."""
    found = set()

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if isinstance(child, ast.Call):
                callee = child.func
                called = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
                if called in LISTING_CALLS:
                    found.add((name, function or "<module>"))
            visit(child, inner)
    visit(tree_of(name), None)
    return found


def test_every_directory_listing_in_the_launch_modules_is_one_that_is_accounted_for():
    """No listing of any folder is added without a deliberate edit of this list: the data tree must never be listed (the
    held-out partition sits beside the files the study reads), and a heuristic on variable names is easy to get around."""
    found = set().union(*(listing_calls(name) for name in LAUNCH_MODULES))
    assert found == LISTING_ALLOWED, sorted(found ^ LISTING_ALLOWED)


@pytest.mark.parametrize("name", LAUNCH_MODULES)
def test_no_launch_module_builds_a_file_name_from_parts(name):
    """A held-out archive name assembled at run time would pass the text search: no concatenation of two string
    constants and no f-string with an `npz` piece."""
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            assert not all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in (node.left, node.right)), (name, node.lineno)
        if isinstance(node, ast.JoinedStr):
            pieces = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
            assert "npz" not in pieces, (name, node.lineno)


def test_production_entry_points_expose_no_switch_for_fixture_dependencies():
    for name in LAUNCH_MODULES:
        source = (PACKAGE / name).read_text()
        assert "LAUNCH_TEST" not in source and "SCENARIO" not in source and "fixture" not in source.lower(), name
    tree = tree_of("launch_stage.py")
    arguments = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                 and n.args and isinstance(n.args[0], ast.Constant)}
    assert not any("fixture" in a or "deps" in a for a in arguments)


def test_the_production_worker_command_line_has_no_tuning_flag():
    from stage2_capped import launch, launch_a0, launch_stage
    parser = launch.build_parser()
    options = set()
    for action in parser._actions:
        for choice in getattr(action, "choices", {}) or {}:
            sub = action.choices[choice] if hasattr(action.choices, "__getitem__") else None
            if sub is not None:
                options |= {o for a in sub._actions for o in a.option_strings}
    options |= {o for a in launch_stage.build_parser()._actions for o in a.option_strings}
    banned = ("ceiling", "storage", "deadline", "stall", "wall", "limit", "output", "launch-id", "ledger", "worker")
    assert not [o for o in options if any(b in o for b in banned)], options
