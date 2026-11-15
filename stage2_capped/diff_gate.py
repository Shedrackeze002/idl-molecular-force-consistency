"""Diff gate: a derived file must equal a frozen file plus a listed set of hunks.

`allowlist.json` names, per target, the frozen file and its SHA-256, the derived
file, and every permitted hunk with its exact removed and added lines. The gate
recomputes the line diff and compares it with the list, so any extra, missing or
altered change fails. The allow-list is a reviewed artifact: its SHA-256 goes
into the launch record, and nothing here regenerates it.

A target may carry `frozen_line_range` [first, last] (1-based, inclusive): the
frozen side of the diff is then those lines of the frozen file (the whole file's
SHA-256 is still pinned, and so is the excerpt's, as `frozen_excerpt_sha256`).
Hunk line numbers are then relative to the excerpt. Targets without it behave
exactly as before.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "allowlist.json"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _lines(data):
    if b"\r" in data:
        raise ValueError("carriage returns are not allowed in gated files")
    return data.decode("utf-8").split("\n")


def compute_hunks(frozen_bytes, derived_bytes):
    """Line-level hunks that turn the frozen text into the derived text."""
    frozen = _lines(frozen_bytes)
    derived = _lines(derived_bytes)
    matcher = difflib.SequenceMatcher(None, frozen, derived, autojunk=False)
    return [{"frozen_first_line": i1 + 1, "frozen_line_count": i2 - i1,
             "removed": frozen[i1:i2], "added": derived[j1:j2]}
            for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def frozen_excerpt(frozen_bytes, line_range):
    """The bytes of lines first..last of the frozen file, newline-terminated."""
    lines = _lines(frozen_bytes)
    first, last = line_range
    if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last <= len(lines)):
        raise ValueError(f"frozen_line_range {line_range!r} is outside the frozen file")
    return ("\n".join(lines[first - 1:last]) + "\n").encode("utf-8")


def check_gate(target, allowlist_path=ALLOWLIST, root=ROOT):
    """Return a list of problems (empty means the gate passes)."""
    entry = json.loads(Path(allowlist_path).read_text())["targets"][target]
    problems = []
    frozen_path = Path(root) / entry["frozen_file"]
    derived_path = Path(root) / entry["derived_file"]
    frozen_bytes = frozen_path.read_bytes()
    derived_bytes = derived_path.read_bytes()
    if sha256_bytes(frozen_bytes) != entry["frozen_sha256"]:
        problems.append(f"frozen file {entry['frozen_file']} does not match its pinned SHA-256")
    if entry.get("frozen_line_range") is not None:
        try:
            frozen_bytes = frozen_excerpt(frozen_bytes, entry["frozen_line_range"])
        except ValueError as error:
            return problems + [str(error)]
        if sha256_bytes(frozen_bytes) != entry.get("frozen_excerpt_sha256"):
            problems.append("the frozen excerpt does not match its pinned SHA-256")
    if frozen_bytes.endswith(b"\n") != derived_bytes.endswith(b"\n"):
        problems.append("trailing newline differs between the frozen and derived file")
    allowed = set(entry["allowed_categories"])
    listed = entry["hunks"]
    actual = compute_hunks(frozen_bytes, derived_bytes)
    for hunk in listed:
        if not hunk["categories"] or not set(hunk["categories"]) <= allowed:
            problems.append(f"listed hunk {hunk['id']} has a category outside the allowed set")
    if len(actual) != len(listed):
        problems.append(f"{len(actual)} hunks found, {len(listed)} listed")
    for index, (found, hunk) in enumerate(zip(actual, listed)):
        for key in ("frozen_first_line", "frozen_line_count", "removed", "added"):
            if found[key] != hunk[key]:
                problems.append(f"hunk {index} ({hunk['id']}) differs from the allow-list in {key}")
                break
    return problems


def assert_gate(target, allowlist_path=ALLOWLIST, root=ROOT):
    problems = check_gate(target, allowlist_path, root)
    if problems:
        raise AssertionError("diff gate failed for %s:\n  %s" % (target, "\n  ".join(problems)))
