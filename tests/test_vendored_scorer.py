"""The vendored scorer is a copy, not a fork.

A reward that has drifted from the thing it scores is not a reward, so the
recorded hashes are checked here and the live Cloris tree is checked too when
it happens to be present — on Colab it is not, which is why the hashes are
recorded in the repo rather than derived on demand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "cloris_boolean_rl" / "scorer"
NOTE = VENDOR / "VENDORED_FROM.json"


@pytest.fixture(scope="module")
def note():
    return json.loads(NOTE.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_note_records_a_commit(note):
    assert len(note["commit"]) == 40
    assert note["source_path"] == "linkedin/boolean_score"


def test_every_vendored_file_matches_its_recorded_hash(note):
    for name, expected in note["sha256"].items():
        assert _digest(VENDOR / name) == expected, f"{name} was edited in this repo"


def test_nothing_was_vendored_that_the_note_does_not_list(note):
    listed = set(note["files"]) | {"VENDORED_FROM.json", "__pycache__"}
    present = {
        str(path.relative_to(VENDOR))
        for path in VENDOR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert present - listed == set()


def test_the_scorer_imports_nothing_outside_itself():
    """What makes it vendorable: stdlib plus its own siblings, nothing else."""
    for name in ("parser.py", "lexicon.py", "score.py", "enrich.py"):
        for line in (VENDOR / name).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "cloris" not in stripped.lower()
                assert not stripped.startswith("from ..")


def test_it_matches_the_live_cloris_tree_byte_for_byte(note):
    source = Path(note["source_repo"]) / note["source_path"]
    if not source.is_dir():
        pytest.skip(f"no Cloris checkout at {source}")
    for name in note["files"]:
        assert _digest(source / name) == _digest(VENDOR / name), (
            f"{name} differs from Cloris; run tools/sync_scorer.py --apply"
        )
