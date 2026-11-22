"""The traceability matrix is complete, names only tests that exist, and matches TRACEABILITY.md."""
from pathlib import Path

from stage2_capped import traceability as tr

MARKDOWN = Path(tr.__file__).with_name("TRACEABILITY.md")


def test_every_section_7_bullet_is_in_the_matrix_once_and_in_order():
    assert [entry["id"] for entry in tr.MATRIX] == list(tr.BULLET_IDS)
    assert len(set(tr.BULLET_IDS)) == len(tr.BULLET_IDS) == 15


def test_every_named_test_exists():
    assert tr.missing_tests() == []


def test_every_bullet_has_tests_and_no_test_is_listed_twice_for_one_bullet():
    for entry in tr.MATRIX:
        assert entry["tests"], entry["id"]
        assert len(set(entry["tests"])) == len(entry["tests"]), entry["id"]
        assert entry["bullet"].strip()


def test_a_bullet_with_a_record_note_is_marked_and_says_why():
    noted = [e for e in tr.MATRIX if e.get("record")]
    assert [e["id"] for e in noted] == ["R1-rules"]
    assert "launch record B" in noted[0]["record"]


def test_the_markdown_file_is_what_render_produces():
    assert MARKDOWN.read_text() == tr.render()


def test_the_matrix_lists_no_file_that_is_not_a_stage2_test_file():
    files = {f for e in tr.MATRIX for f, _ in e["tests"]}
    assert files <= {p.name for p in tr.TESTS.glob("test_*.py")}


def test_the_plan_m6_items_are_listed_and_covered():
    assert [i["id"] for i in tr.PLAN_ITEMS] == ["M6-traceability", "M6-frozen-files", "M6-dry-run",
                                                    "M4-review-fixes"]
    assert all(i["tests"] for i in tr.PLAN_ITEMS)
