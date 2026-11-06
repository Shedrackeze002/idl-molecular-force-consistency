"""Amendment A5: verify_launch is read only and discloses every dispatch, including failed and refused ones."""
import json

from stage2_capped import launch_io, launch_record, launch_state, verify_launch
from stage2_capped.launch_state import EventLog, Registry
from stage2_capped.tests.launch_fixtures import make_stage_directory

SHA_A, SHA_B = "a" * 64, "b" * 64
SHA_A2, SHA_B2 = "c" * 64, "d" * 64
RECORD = {"saved_file_hashes": {}}


def write_records(folder, specs):
    (folder / "records").mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, body in specs.items():
        data = launch_state.dumps(body)
        (folder / "records" / name).write_bytes(data)
        shas[name] = launch_record.sha256_bytes(data)
    return shas


def build(tmp_path):
    folder = tmp_path / "launch"
    roots = {key: tmp_path / key for key in ("a", "b", "a2", "b2")}
    for key, root in roots.items():
        launch_io.fresh_root(root, key)
    shas = write_records(folder, {
        "RECORD_A.json": {"record": "A", "output_root": str(roots["a"])},
        "RECORD_B.json": {"record": "B", "output_root": str(roots["b"]), "record_a_sha256": None},
        "RECORD_A2.json": {"record": "A", "output_root": str(roots["a2"])},
        "RECORD_B2.json": {"record": "B", "output_root": str(roots["b2"]), "record_a_sha256": None}})
    # pair records by rewriting B's with the real A hashes
    shas.update(write_b(folder, roots, shas))
    return folder, roots, shas, Registry(folder / "registry")


def write_b(folder, roots, shas):
    out = {}
    for name, key, a in (("RECORD_B.json", "b", "RECORD_A.json"), ("RECORD_B2.json", "b2", "RECORD_A2.json")):
        data = launch_state.dumps({"record": "B", "output_root": str(roots[key]), "record_a_sha256": shas[a]})
        (folder / "records" / name).write_bytes(data)
        out[name] = launch_record.sha256_bytes(data)
    return out


def dispatch(registry, sha, unit, attempt, outcome, spent=True, evidence=None):
    registry.write(sha, unit, attempt, "started", launch_id="x")
    if outcome is not None:
        registry.write(sha, unit, attempt, "exited", outcome=outcome, spent=spent, evidence=evidence or {})


def test_every_dispatch_is_listed_including_refused_failed_and_unfinished_ones(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    dispatch(registry, shas["RECORD_A.json"], "A0", 1, "refused_before_start", spent=False)
    dispatch(registry, shas["RECORD_A.json"], "A0", 2, "failed")
    dispatch(registry, shas["RECORD_A2.json"], "A0", 1, None)
    lines, problems = verify_launch.verify(folder)
    text = "\n".join(lines)
    assert lines[0] == "3 dispatches in the registry for this study"
    assert "attempt 1: refused_before_start; spent False" in text and "attempt 2: failed; spent True" in text
    assert "started, no exited event (counts as spent)" in text
    assert problems == []


def test_a_completed_unit_says_whether_it_completed_in_the_first_record_pair(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    b, b2 = shas["RECORD_B.json"], shas["RECORD_B2.json"]
    directory, marker = make_stage_directory(roots["b"], "0b", RECORD, "b", "r" * 64)
    dispatch(registry, b, "0b", 1, "failed")
    directory2, marker2 = make_stage_directory(roots["b2"], "0b", RECORD, "b2", "r" * 64)
    dispatch(registry, b2, "0b", 1, "complete", evidence={"STAGE_COMPLETE.json": marker2})
    dispatch(registry, shas["RECORD_A.json"], "A0", 1, "complete")
    lines, problems = verify_launch.verify(folder)
    text = "\n".join(lines)
    assert "unit 0b completed in a later record pair" in text and "unit A0 completed in the first record pair" in text
    assert problems == []


def test_a_recorded_hash_that_differs_from_the_file_or_a_missing_file_is_a_problem(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    b = shas["RECORD_B.json"]
    make_stage_directory(roots["b"], "0b", RECORD, "b", "r" * 64)
    dispatch(registry, b, "0b", 1, "complete", evidence={"STAGE_COMPLETE.json": "0" * 64})
    dispatch(registry, b, "1", 1, "complete", evidence={"STAGE_COMPLETE.json": "1" * 64})
    _, problems = verify_launch.verify(folder)
    assert any("STAGE_COMPLETE.json differs" in p for p in problems) and any("missing" in p for p in problems)


def test_a_broken_event_chain_a_missing_root_and_a_violation_file_are_problems(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    log = EventLog(roots["b"])
    for name in ("a", "b"):
        log.append(name)
    first = log.files()[0]
    first.write_text(first.read_text().replace('"n": 1', '"n": 1, "x": 0'))
    (folder / "violations").mkdir()
    (folder / "violations" / f"AUDIT_VIOLATION-{shas['RECORD_A.json']}-A0-1.json").write_text("{}")
    dispatch(registry, shas["RECORD_A2.json"], "A0", 1, "failed")
    roots["a2"].rename(tmp_path / "a2-moved")
    _, problems = verify_launch.verify(folder)
    assert any("the chain is broken" in p for p in problems)
    assert any("audit violation file" in p for p in problems)
    assert any("output root" in p and "does not exist" in p for p in problems)


def test_a_receipt_is_compared_with_the_registry(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    b = shas["RECORD_B.json"]
    make_stage_directory(roots["b"], "0", RECORD, "b", "r" * 64)
    marker = launch_record.sha256_bytes((roots["b"] / "rows" / "stage0" / "STAGE_COMPLETE.json").read_bytes())
    dispatch(registry, b, "0", 1, "complete", evidence={"STAGE_COMPLETE.json": marker})
    good = tmp_path / "receipt-good.json"
    good.write_text(json.dumps({"stage_marker_sha256": {"0": marker}}))
    assert verify_launch.verify(folder, good)[1] == []
    bad = tmp_path / "receipt-bad.json"
    bad.write_text(json.dumps({"stage_marker_sha256": {"0": "0" * 64}, "a0_result_sha256": "1" * 64}))
    problems = verify_launch.verify(folder, bad)[1]
    assert len(problems) == 2


def test_verification_writes_nothing(tmp_path):
    folder, roots, shas, registry = build(tmp_path)
    dispatch(registry, shas["RECORD_A.json"], "A0", 1, "failed")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    verify_launch.verify(folder)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_an_empty_folder_has_no_dispatches_and_no_problems(tmp_path):
    lines, problems = verify_launch.verify(tmp_path / "nothing")
    assert lines[0] == "0 dispatches in the registry for this study" and problems == []


def test_the_command_line_prints_problems_and_exits_nonzero(tmp_path, capsys):
    folder, roots, shas, registry = build(tmp_path)
    make_stage_directory(roots["b"], "0b", RECORD, "b", "r" * 64)
    dispatch(registry, shas["RECORD_B.json"], "0b", 1, "complete", evidence={"STAGE_COMPLETE.json": "0" * 64})
    assert verify_launch.main(["--launch-folder", str(folder)]) == 1
    assert "problem:" in capsys.readouterr().out
