"""The prompt set: no leaks, no collisions, and a contract the reward reads.

The single failure this file exists to prevent is a held-out concept showing
up as a training target. Everything else here is scaffolding around that.
"""

from __future__ import annotations

import pytest

from cloris_boolean_rl.data import prompts as prompts_mod
from cloris_boolean_rl.data import split as split_mod
from cloris_boolean_rl.reward import tasks


@pytest.fixture(scope="module")
def plan():
    return split_mod.load_split()


@pytest.fixture(scope="module")
def built():
    return prompts_mod.build_prompts()


def test_the_written_set_matches_the_built_one(built):
    saved = prompts_mod.load_prompts()
    assert [p.id for p in saved] == [p.id for p in built]
    assert [p.given for p in saved] == [p.given for p in built]


def test_prompt_ids_are_unique(built):
    ids = [p.id for p in built]
    assert len(set(ids)) == len(ids)


def test_no_training_prompt_targets_a_held_out_concept(built, plan):
    held = set(plan.b_ids)
    for prompt in built:
        if prompt.split == prompts_mod.TRAIN:
            assert not (set(prompt.targets) & held), prompt.id


def test_no_eval_prompt_targets_a_trained_concept(built, plan):
    trained = set(plan.a_ids)
    for prompt in built:
        if prompt.split == prompts_mod.EVAL:
            assert not (set(prompt.targets) & trained), prompt.id


def test_no_held_out_surface_appears_in_a_training_prompts_text(built, plan):
    """The stronger claim: B's WORDS are absent from what the policy reads.

    Disjoint concept ids would still leak if a held-out spelling turned up
    inside a training prompt's given text, because the policy would have read
    it before the eval measured it.
    """
    document = split_mod.load_full_lexicon_document()
    b_only = split_mod.surfaces_for(document, plan.b_ids) - split_mod.surfaces_for(
        document, plan.a_ids
    )
    leaks = []
    for prompt in built:
        if prompt.split != prompts_mod.TRAIN:
            continue
        seen = {term.normalized for term in tasks._terms_of(prompt.given)}
        seen |= {term.normalized for term in tasks._terms_of(prompt.reference)}
        seen |= {prompt.given.strip('"').lower()}
        overlap = seen & b_only
        if overlap:
            leaks.append((prompt.id, sorted(overlap)))
    assert leaks == []


def test_every_prompt_carries_a_task_and_a_split(built):
    for prompt in built:
        assert prompt.task in (prompts_mod.REGISTER, prompts_mod.ANCHOR)
        assert prompt.split in (prompts_mod.TRAIN, prompts_mod.EVAL)
        assert prompt.given.strip()
        assert prompt.targets


def test_every_given_is_a_legal_group_or_a_quoted_term(built):
    for prompt in built:
        text = prompt.given
        if text.startswith("("):
            assert tasks.extract_group(text) is not None, prompt.id
        else:
            assert text.startswith('"') and text.endswith('"'), prompt.id


def test_every_reference_satisfies_the_output_contract(built):
    for prompt in built:
        assert tasks.extract_group(prompt.reference) is not None, prompt.id


def test_the_doctrine_states_the_output_contract(built):
    for task in (prompts_mod.REGISTER, prompts_mod.ANCHOR):
        system = prompts_mod.Prompt(
            id="x", task=task, split="train", source="t", given='"x"', targets=("x",)
        ).system
        assert "OUTPUT CONTRACT" in system
        assert "ONE parenthesised OR group" in system
        assert len(system.splitlines()) < 60


def test_there_are_enough_prompts_for_a_first_run(built):
    tally = prompts_mod.counts(built)
    assert tally["train/register"] >= 100
    assert tally["train/anchor"] >= 25
    assert tally["eval/register"] >= 25
    assert tally["eval/anchor"] >= 15


def test_selection_is_deterministic_and_balanced(built):
    first = prompts_mod.select(built, split="train", limit=100)
    second = prompts_mod.select(built, split="train", limit=100)
    assert [p.id for p in first] == [p.id for p in second]
    assert len(first) == 100
    tasks_seen = {p.task for p in first}
    assert tasks_seen == {prompts_mod.REGISTER, prompts_mod.ANCHOR}
