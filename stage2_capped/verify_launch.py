"""Read-only verification of a launch from its records, registry and roots (plan v3 section 9; amendment A5).

    python -m stage2_capped.verify_launch --launch-folder FOLDER [--receipt RECORD_B_RECEIPT.json]

Report authors run it before any number is used. It writes nothing. It prints every dispatch in the registry for
the study (record SHA-256, unit, attempt, outcome, hash of the marker), including failed and refused ones, and for
each completed unit says whether it completed in the first record pair; it re-derives each output root's event
chain; it compares every `exited` event's recorded file hashes with the files in the root; it lists audit violation
files; and, if a receipt is given, compares the typed hashes it holds with the markers. Exit 0 if no problem was
found, 1 otherwise. The record SHA-256 of a file is the SHA-256 of its bytes.
"""
import argparse
import json
import sys
from pathlib import Path

from stage2_capped import launch_record, launch_state


def load_records(folder):
    records = {}
    directory = Path(folder) / "records"
    for path in sorted(directory.glob("RECORD_*.json")) if directory.is_dir() else []:
        if path.name.endswith("RECEIPT.json"):
            continue
        data = path.read_bytes()
        try:
            records[launch_record.sha256_bytes(data)] = {"file": path.name, **json.loads(data)}
        except ValueError:
            continue
    return records


def pairs_of(records):
    """{record B sha: record A sha} from record B's own field; A shas without a B stand alone."""
    pairs = {sha: record["record_a_sha256"] for sha, record in records.items() if record.get("record") == "B"}
    return pairs


def marker_hash(evidence):
    return (evidence or {}).get("STAGE_COMPLETE.json") or (evidence or {}).get("A0_RESULT.json") or "none"


def verify(launch_folder, receipt=None):
    """(lines, problems). Reads the registry, the records and the roots; never writes."""
    folder = Path(launch_folder)
    registry = launch_state.Registry(folder / "registry")
    records = load_records(folder)
    pairs = pairs_of(records)
    entries = registry.entries()
    lines, problems = [], []
    first_start = {}
    for entry in entries:
        if entry["event"] == "started":
            first_start[entry["record_sha256"]] = min(first_start.get(entry["record_sha256"], float("inf")),
                                                      entry["payload"].get("time", float("inf")))

    def pair_key(sha):
        return sha if sha in pairs else next((b for b, a in pairs.items() if a == sha), sha)
    pair_time = {}
    for sha, started in first_start.items():
        key = pair_key(sha)
        pair_time[key] = min(pair_time.get(key, float("inf")), started)
    first_pair = min(pair_time, key=pair_time.get) if pair_time else None
    attempts = sorted({(e["record_sha256"], e["unit"], e["attempt"]) for e in entries},
                      key=lambda item: (first_start.get(item[0], 0), item[1], item[2]))
    lines.append(f"{len(attempts)} dispatches in the registry for this study")
    for sha, unit, number in attempts:
        pair = registry.attempts(sha, unit)[number]
        exited = pair["exited"]["payload"] if pair["exited"] else None
        outcome = exited["outcome"] if exited else "started, no exited event (counts as spent)"
        spent = (exited.get("spent") if exited else True)
        kind = records.get(sha, {}).get("record", "?")
        lines.append(f"dispatch record {kind} {sha[:16]} unit {unit} attempt {number}: {outcome}; spent {spent}; "
                     f"marker {marker_hash((exited or {}).get('evidence'))}")
        if exited and exited.get("outcome") == "complete":
            in_first = pair_key(sha) == first_pair
            lines.append(f"  unit {unit} completed in {'the first' if in_first else 'a later'} record pair")
    # Roots: the event chain and the recorded file hashes.
    for sha, record in records.items():
        root = Path(record["output_root"]) if record.get("output_root") else None
        if root is None or not root.is_dir():
            if any(e["record_sha256"] == sha for e in entries):
                problems.append(f"record {record['file']}: its output root {record.get('output_root')} does not exist")
            continue
        problems += [f"{root.name}: {p}" for p in launch_state.EventLog(root).problems()]
        for entry in entries:
            if entry["record_sha256"] != sha or entry["event"] != "exited":
                continue
            for name, digest in (entry["payload"].get("evidence") or {}).items():
                path = root / name if name.startswith("A0_") else (
                    launch_state.stage_directory(root, entry["unit"]) / name if name == "STAGE_COMPLETE.json" else root / name)
                if not path.is_file():
                    problems.append(f"{entry['file']}: {name} is recorded but missing from {root.name}")
                elif launch_record.sha256_bytes(path.read_bytes()) != digest:
                    problems.append(f"{entry['file']}: {name} differs from its recorded hash")
    for name in launch_state.violation_files(folder, set(records)):
        problems.append(f"audit violation file {name}: the unit is void and the record pair unusable")
    if receipt is not None:
        problems += receipt_problems(json.loads(Path(receipt).read_text()), records, registry)
    lines.append("problems: " + ("none" if not problems else str(len(problems))))
    return lines, problems


def receipt_problems(receipt, records, registry):
    """The receipt's stage marker hashes (and the A0 result hash) against the exited events."""
    problems = []
    recorded = {}
    for entry in registry.entries():
        if entry["event"] == "exited" and entry["payload"].get("outcome") == "complete":
            recorded[entry["unit"]] = marker_hash(entry["payload"].get("evidence"))
    for stage, digest in (receipt.get("stage_marker_sha256") or {}).items():
        if recorded.get(stage) != digest:
            problems.append(f"the receipt's stage {stage} marker hash is not the one the registry recorded")
    if receipt.get("a0_result_sha256") is not None and recorded.get("A0") != receipt["a0_result_sha256"]:
        problems.append("the receipt's A0 result hash is not the one the registry recorded")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a launch from its records, registry and roots (read only)")
    parser.add_argument("--launch-folder", required=True)
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args(argv)
    lines, problems = verify(args.launch_folder, args.receipt)
    print("\n".join(lines + [f"problem: {p}" for p in problems]))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
