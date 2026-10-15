"""The verifiable reward: the scorer, plus the gates the scorer does not have.

The scorer is the judge. Nothing here re-implements it, re-weights it, or
softens it — the reward for a completion is ``ScoreReport.log_composite``, in
log space because the composite is a product and GRPO normalises advantages
within a group, so a multiplicative reward would make every improvement on a
bad completion invisible.

What IS here is the set of refusals the scorer was never built to make,
because the scorer judges strings a person wrote and this judges strings a
policy is being paid to write. Three differences matter.

A completion has an OUTPUT CONTRACT. One parenthesised OR group, nothing else.
The scorer happily reads a two-gate string, so a policy asked for one group
and paid by the scorer would learn to answer with three. Anything that is not
exactly one positive group floors.

A completion has a TARGET. The prompt names a concept, or a sweep; the scorer
does not know that and would pay full marks for a beautifully expanded group
about something else. So a register completion that names none of the prompt's
concepts floors, and an anchor completion whose family does not reach the
sweep's floors.

A completion can be PADDED, and this is the one the scorer genuinely gets
wrong. Measured against the A lexicon, a sixty-term dump of common surfaces
composites at 0.514 — above the 0.413 of a bare, honest term — because
``_parsimony`` charges rare forms, grab-bags, wasted gates and length in
characters, but nothing at all for the number of terms in a group. On a
coverage-weighted reward, "say every word you know" is therefore a rising
gradient, and it is the first thing a GRPO policy finds. The focus gate is
the fix: a completion may name at most a handful of concepts, all of them
sharing a family with the prompt's target, and at most sixteen terms. Both
numbers sit well above anything a real answer needs — the richest concept in
the lexicon has nine non-rare forms and a sweep saturates at fifteen — so the
gate refuses the hack without touching the behaviour.

Every floor is the scorer's own zero, ``log(1e-6)``, rather than an invented
constant: an illegal string already scores exactly that, so a refused
completion and a broken one are worth the same and the reward has one scale.
For the anchor task, whose reward is a delta, the floor is that same zero
minus the sweep's own score — the truthful statement that the proposal made
the string worthless.

A register answer that is legal but about the wrong concept (off-target or
off-concept) is not floored flat. It is scored like an answer and paid a
quarter of the way from the floor to that score, because GRPO learns only
from the spread inside a group: at a flat floor a legal wrong group was worth
exactly what prose was, and the first run's off-target answers doubled. A
contract failure or a padded dump stays at the flat floor, because those are
the hacks the floor exists to refuse and grading them would pay for them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from ..data.prompts import ANCHOR, EVAL, REGISTER, TRAIN, Prompt
from ..scorer.lexicon import Lexicon
from ..scorer.parser import Not, Or, Term, contains_and, parse
from ..scorer.score import LOG_FLOOR, _groups_for, _is_anchor, score, shape_of
from . import lexicons

FLOOR = math.log(LOG_FLOOR)

# Generous ceilings. The richest concept in the lexicon declares nine non-rare
# forms; a sweep saturates at fifteen named things. Anything past these is not
# a group a sourcer would write, it is a word list.
MAX_TERMS = 16
MAX_CONCEPTS = 5

# The share of the way from the floor to its own score that a legal register
# group about the wrong concept earns. High enough that "legal but wrong" sits
# clearly above prose in a GRPO group, low enough that every on-target answer
# whose composite clears 1e-4.5 (about 3e-5) still beats every wrong one.
OFF_TARGET_CREDIT = 0.25

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


@dataclass(frozen=True)
class RewardOutcome:
    """What the reward decided, and on what grounds.

    Kept so the eval can report WHY a checkpoint floors — a policy floored for
    prose is a formatting problem and a policy floored for padding is a reward
    hack, and the two want opposite responses.
    """

    value: float
    gate: str
    group: str = ""
    combined: float = 0.0
    baseline: float = 0.0
    composite: float = 0.0
    shape: str = ""


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------


def extract_group(completion: str) -> str | None:
    """The single parenthesised OR group a completion promised, or ``None``.

    Only whitespace and a code fence are forgiven. A trailing sentence, a
    second group and a stray operator are not: the contract is in the prompt,
    and a reward that quietly repairs the output teaches the policy that the
    contract is optional.

    Parser WARNINGS are deliberately let through. An unquoted token or a
    lowercase operator is already charged by the scorer's structure term,
    which multiplies into parsimony, so refusing them here would charge the
    same defect twice — and it would flatten exactly the near-misses a cold
    policy produces into the same floor as prose. GRPO learns from the spread
    inside the group; a gate that collapses every imperfect answer to one
    value hands it a gradient of zero.
    """
    if not isinstance(completion, str):
        return None
    text = _FENCE_RE.sub("", completion).strip()
    if not text.startswith("(") or not text.endswith(")"):
        return None
    result = parse(text)
    if not result.legal or result.tree is None:
        return None
    if not isinstance(result.tree, (Or, Term)):
        # The root is an And or a Not: two gates, not one group.
        return None
    if contains_and(result.tree) or _contains_not(result.tree):
        # Nested, as in ("a b" written unquoted) or (x OR (y AND z)).
        return None
    return text


def _contains_not(node) -> bool:
    if isinstance(node, Not):
        return True
    if isinstance(node, Or):
        return any(_contains_not(child) for child in node.children)
    return False


def _terms_of(text: str) -> tuple[Term, ...]:
    tree = parse(text).tree
    if isinstance(tree, Term):
        return (tree,)
    if isinstance(tree, Or):
        return tuple(child for child in tree.children if isinstance(child, Term))
    return ()


def _direct_families(lexicon: Lexicon, concept_id: str) -> frozenset[str]:
    for concept in lexicon.concepts:
        if concept.id == concept_id:
            return frozenset(concept.families)
    return frozenset()


def _mapped_ids(lexicon: Lexicon, text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            concept.id
            for term in _terms_of(text)
            if (concept := lexicon.concept_for(term.text)) is not None
        )
    )


# ---------------------------------------------------------------------------
# Task 1: register generalisation
# ---------------------------------------------------------------------------


def explain_task1(prompt: Prompt, completion: str) -> RewardOutcome:
    group = extract_group(completion)
    if group is None:
        return RewardOutcome(FLOOR, "contract")

    terms = _terms_of(group)
    if not terms:
        return RewardOutcome(FLOOR, "contract")
    if len(terms) > MAX_TERMS:
        return RewardOutcome(FLOOR, "padded", group)

    gate = lexicons.gate_lexicon(prompt.split)
    target_families: set[str] = set()
    for concept_id in prompt.targets:
        target_families |= _direct_families(gate, concept_id)

    named = _mapped_ids(gate, group)
    if len(named) > max(MAX_CONCEPTS, 2 * len(prompt.targets)):
        return RewardOutcome(FLOOR, "padded", group)
    lexicon = lexicons.reward_lexicon(prompt.split, REGISTER)
    if any(
        not (_direct_families(gate, concept_id) & target_families)
        for concept_id in named
    ):
        return _graded_miss(group, "off_concept", lexicon)
    if not set(named) & set(prompt.targets):
        # Nothing the prompt asked about is in the answer. Either the policy
        # ignored the prompt or it answered in words the lexicon cannot read,
        # and neither is a completion this reward can vouch for as an answer.
        return _graded_miss(group, "off_target", lexicon)

    report = score(group, lexicon)
    return RewardOutcome(
        report.log_composite,
        "ok",
        group,
        composite=report.composite,
        shape=report.shape,
    )


def _graded_miss(group: str, gate: str, lexicon: Lexicon) -> RewardOutcome:
    """A legal group about the wrong concept: a quarter of the way up.

    The group's own log composite is taken at most 0, so the credit is capped
    at ``FLOOR * (1 - OFF_TARGET_CREDIT)``, and at least the floor.
    """
    report = score(group, lexicon)
    s = min(report.log_composite, 0.0)
    value = max(FLOOR, FLOOR + OFF_TARGET_CREDIT * (s - FLOOR))
    return RewardOutcome(
        value, gate, group, composite=report.composite, shape=report.shape
    )


def reward_task1(prompt: Prompt, completion: str) -> float:
    return explain_task1(prompt, completion).value


# ---------------------------------------------------------------------------
# Task 2: anchor proposal
# ---------------------------------------------------------------------------


def sweep_baseline(prompt: Prompt) -> float:
    lexicon = lexicons.reward_lexicon(prompt.split, ANCHOR)
    return score(prompt.given, lexicon).log_composite


def reward_floor_for(prompt: Prompt) -> float:
    """The value a refused completion is worth for this prompt."""
    if prompt.task == REGISTER:
        return FLOOR
    return FLOOR - sweep_baseline(prompt)


def explain_task2(prompt: Prompt, completion: str) -> RewardOutcome:
    baseline = sweep_baseline(prompt)
    floor = FLOOR - baseline

    group = extract_group(completion)
    if group is None:
        return RewardOutcome(floor, "contract", baseline=baseline)

    terms = _terms_of(group)
    if not terms or len(terms) > MAX_TERMS:
        return RewardOutcome(floor, "padded", group, baseline=baseline)

    gate = lexicons.gate_lexicon(prompt.split)

    # An anchor is general terms. A group made only of words every profile
    # carries is not general, it is empty.
    if all(gate.is_ubiquitous(term.text) for term in terms):
        return RewardOutcome(floor, "umbrella_only", group, baseline=baseline)

    named = _mapped_ids(gate, group)
    if not named:
        return RewardOutcome(floor, "unreadable", group, baseline=baseline)
    concepts = {concept.id: concept for concept in gate.concepts}
    if any(not concepts[cid].names_a_class for cid in named):
        return RewardOutcome(floor, "not_all_class", group, baseline=baseline)

    shared = set(_direct_families(gate, named[0]))
    for cid in named[1:]:
        shared &= _direct_families(gate, cid)
    if not shared:
        return RewardOutcome(floor, "multi_family", group, baseline=baseline)

    # The anchor has to be the right class for THIS sweep, or a policy that
    # memorised one good anchor would answer every prompt with it.
    sweep_reach: set[str] = set()
    for cid in _mapped_ids(gate, prompt.given):
        sweep_reach |= set(gate.family_reach(cid))
    sweep_reach |= set(prompt.families)
    anchor_reach: set[str] = set()
    for cid in named:
        anchor_reach |= set(gate.family_reach(cid))
    if not (anchor_reach & sweep_reach):
        return RewardOutcome(floor, "wrong_family", group, baseline=baseline)

    # Repeating the sweep's own terms is not proposing an anchor.
    sweep_surfaces = {term.normalized for term in _terms_of(prompt.given)}
    if {term.normalized for term in terms} & sweep_surfaces:
        return RewardOutcome(floor, "repeats_sweep", group, baseline=baseline)

    lexicon = lexicons.reward_lexicon(prompt.split, ANCHOR)
    combined_text = f"{group} AND {prompt.given}"
    result = parse(combined_text)
    if not result.legal:
        return RewardOutcome(floor, "contract", group, baseline=baseline)
    groups = _groups_for(result.tree, lexicon)
    shape = shape_of(groups, lexicon, anchored_by_filter=False)
    if shape != "anchor_and_sweep":
        return RewardOutcome(floor, "not_anchored", group, baseline=baseline, shape=shape)
    proposed = next(
        (g for g in groups if not g.negated and g.surfaces <= {t.normalized for t in terms}),
        None,
    )
    if proposed is None or not _is_anchor(proposed, lexicon):
        return RewardOutcome(floor, "not_anchored", group, baseline=baseline, shape=shape)

    report = score(combined_text, lexicon)
    return RewardOutcome(
        report.log_composite - baseline,
        "ok",
        group,
        combined=report.log_composite,
        baseline=baseline,
        composite=report.composite,
        shape=report.shape,
    )


def reward_task2(prompt: Prompt, completion: str) -> float:
    return explain_task2(prompt, completion).value


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def explain(prompt: Prompt, completion: str) -> RewardOutcome:
    if prompt.task == REGISTER:
        return explain_task1(prompt, completion)
    return explain_task2(prompt, completion)


def reward(prompt: Prompt, completion: str) -> float:
    return explain(prompt, completion).value


def rewards(prompts: Sequence[Prompt], completions: Sequence[str]) -> list[float]:
    return [reward(prompt, text) for prompt, text in zip(prompts, completions)]
