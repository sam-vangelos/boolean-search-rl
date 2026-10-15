"""The A/B concept split: deterministic, disjoint, and stratified.

The split is the load-bearing claim of Stage 3. If B leaks into A, a policy
that memorised the word list passes the eval, and the eval says nothing.
"""

from __future__ import annotations

import json

import pytest

from cloris_boolean_rl.data import split as split_mod
from cloris_boolean_rl.scorer.lexicon import build_lexicon


@pytest.fixture(scope="module")
def full():
    return split_mod.load_full_lexicon_document()


@pytest.fixture(scope="module")
def plan(full):
    return split_mod.build_split(full)


def test_split_is_deterministic(full):
    first = split_mod.build_split(full)
    second = split_mod.build_split(full)
    assert first.a_ids == second.a_ids
    assert first.b_ids == second.b_ids


def test_a_and_b_are_disjoint_and_exhaustive(full, plan):
    declared = {concept["id"] for concept in full["concepts"]}
    assert set(plan.a_ids) & set(plan.b_ids) == set()
    assert set(plan.a_ids) | set(plan.b_ids) == declared


def test_b_is_about_a_quarter(plan):
    share = len(plan.b_ids) / (len(plan.a_ids) + len(plan.b_ids))
    assert 0.18 <= share <= 0.32


def test_every_multi_concept_family_has_a_held_out_concept(full, plan):
    held = set(plan.b_ids)
    for family, ids in split_mod.eligible_by_family(full).items():
        if len(ids) >= 2:
            assert held & set(ids), f"family {family!r} has no held-out concept"


def test_every_family_keeps_a_concept_in_a(full, plan):
    kept = set(plan.a_ids)
    for family, ids in split_mod.concepts_by_family(full).items():
        assert kept & set(ids), f"family {family!r} was emptied into B"


def test_umbrella_and_generic_concepts_are_never_held_out(full, plan):
    protected = split_mod.protected_ids(full)
    assert protected, "expected the umbrella/generic concepts to be identified"
    assert not (protected & set(plan.b_ids))


def test_both_sub_lexicons_load(full, plan):
    lex_a = build_lexicon(split_mod.subset_document(full, plan.a_ids))
    lex_b = build_lexicon(split_mod.subset_document(full, plan.b_ids))
    assert {concept.id for concept in lex_a.concepts} == set(plan.a_ids)
    assert {concept.id for concept in lex_b.concepts} == set(plan.b_ids)


def test_a_lexicon_cannot_read_any_b_surface(full, plan):
    lex_a = build_lexicon(split_mod.subset_document(full, plan.a_ids))
    b_only = split_mod.surfaces_for(full, plan.b_ids) - split_mod.surfaces_for(
        full, plan.a_ids
    )
    assert b_only
    leaked = sorted(term for term in b_only if lex_a.concept_for(term) is not None)
    assert leaked == []


def test_written_split_matches_the_computed_one(plan):
    path = split_mod.SPLIT_PATH
    assert path.exists(), "run tools/build_split.py"
    saved = json.loads(path.read_text())
    assert saved["a"] == list(plan.a_ids)
    assert saved["b"] == list(plan.b_ids)
    assert saved["seed"] == split_mod.SEED
