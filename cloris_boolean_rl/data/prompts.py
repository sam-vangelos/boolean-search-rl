"""Build the training and evaluation prompt sets.

Two tasks, because Stage 2's measurement left exactly two things a trained
policy could add to a deterministic enricher that already saturates the
mechanical defects.

REGISTER (``task="register"``). Given a bare term or a thin group, emit the OR
group a good sourcer would write. The enricher does this perfectly for terms
its lexicon lists and cannot do it at all otherwise, because it invents
nothing. The interesting case is therefore the unlisted concept, which is what
the B split manufactures.

ANCHOR (``task="anchor"``). Given a sweep of named things with nothing general
to pin it, propose the class group to AND with it. No amount of adding
spellings creates an anchor — Stage 2's residue table says so in numbers:
missing_anchor is the largest remaining defect on every corpus it measured.

Prompts come from two places. The lexicon supplies clean, controlled cases:
one concept, its forms known, so the reference answer is exactly checkable.
The real corpora — the 76 calibration strings and the 491 strings Cloris
actually emitted — supply the messy ones, where a group carries three concepts
and half its terms are not in any lexicon.

The split rule is absolute and is enforced here rather than trusted: a TRAIN
prompt's target concepts all come from A, an EVAL prompt's all come from B,
and a prompt whose targets straddle the split is dropped rather than assigned.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..scorer.enrich import enrich
from ..scorer.lexicon import Concept, Lexicon, build_lexicon, normalize
from ..scorer.parser import parse
from ..scorer.score import _groups_for, shape_of
from . import split as split_mod
from .doctrine import system_prompt, user_prompt

# Not "prompts/": a data directory beside prompts.py with the same stem
# resolves as a namespace package and can shadow the module.
PROMPT_DIR = split_mod.DATA_DIR / "promptsets"
PROMPT_PATH = PROMPT_DIR / "re-v1.json"

TRAIN = "train"
EVAL = "eval"
REGISTER = "register"
ANCHOR = "anchor"

SEED = 20260921
# A concept with one spelling has no register to cover, so asking for its
# group teaches nothing and scores the same however the policy answers.
MIN_FORMS_FOR_REGISTER = 2
# A sweep of one named thing is not a sweep.
MIN_SWEEP_TERMS = 2
MAX_SWEEP_TERMS = 6


@dataclass(frozen=True)
class Prompt:
    """One training or evaluation item.

    ``given`` is what the policy is shown; ``reference`` is the best answer the
    lexicon knows, kept for the eval's ceiling and never shown to the policy.
    """

    id: str
    task: str
    split: str
    source: str
    given: str
    targets: tuple[str, ...]
    reference: str = ""
    families: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def system(self) -> str:
        return system_prompt(self.task)

    @property
    def user(self) -> str:
        return user_prompt(self.task, self.given)

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "split": self.split,
            "source": self.source,
            "given": self.given,
            "targets": list(self.targets),
            "families": list(self.families),
            "reference": self.reference,
            "meta": self.meta,
        }


def prompt_from_dict(row: dict) -> Prompt:
    return Prompt(
        id=row["id"],
        task=row["task"],
        split=row["split"],
        source=row["source"],
        given=row["given"],
        targets=tuple(row["targets"]),
        reference=row.get("reference", ""),
        families=tuple(row.get("families", ())),
        meta=dict(row.get("meta") or {}),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_group(terms: Iterable[str]) -> str:
    seen: list[str] = []
    for term in terms:
        cleaned = " ".join(str(term).split()).strip('"')
        if cleaned and cleaned.lower() not in {item.lower() for item in seen}:
            seen.append(cleaned)
    return "(" + " OR ".join(f'"{term}"' for term in seen) + ")"


def _usable_forms(concept: Concept) -> list[str]:
    """The forms a good group is expected to carry: common, then occasional.

    Rare forms are left out on purpose. They are what the enricher deletes and
    what parsimony charges, so putting them in a reference answer would make
    the reference worse than the thing it is the reference for.
    """
    common = [form.text for form in concept.forms if form.register == "common"]
    occasional = [form.text for form in concept.forms if form.register == "occasional"]
    return common + occasional


# The reference anchor has to pass the same gate the policy's answer does —
# all class concepts, one shared family — or the ceiling sits below the floor
# and the eval reads as if nothing can win.
MAX_ANCHOR_CONCEPTS = 2
MAX_ANCHOR_TERMS = 8


def finish(group: str, lexicon: Lexicon) -> str:
    """The group with its twins added, by the deterministic enricher.

    A reference has to be the best answer, and a list of a concept's declared
    forms is not quite that: the hyphen twin is derived rather than declared,
    so ``("LoRA" OR "low-rank adaptation")`` is missing ``"low rank
    adaptation"`` and scores BELOW the bare ``("LoRA")`` on the twin axis —
    measured, -0.393 against -0.322. Running the enricher over the reference
    closes that, and it also screens the twins that would trip the scorer's
    own structure checks, which hand-assembling them would not.
    """
    report = enrich(group, lexicon)
    return report.enriched if report.legal and report.enriched else group


def anchor_reference(
    lexicon: Lexicon, family: str
) -> tuple[str, tuple[str, ...]]:
    """The class group that anchors this family, or ``("", ())``.

    Seven of the lexicon's twenty-three families — datasets, model names,
    training frameworks, app frameworks, venues, vector stores, cloud
    platforms — declare no class concept and sit under no parent that does.
    A sweep of their members has no anchor the scorer can read, so it is not
    a prompt: the reward would floor every answer, correct ones included.
    That is a gap in the lexicon, reported rather than papered over.
    """
    candidates = [
        concept
        for concept in lexicon.concepts
        if concept.names_a_class and family in concept.families
    ]
    if not candidates:
        return "", ()
    candidates.sort(key=lambda c: (-len(_usable_forms(c)), c.id))
    picked = candidates[:MAX_ANCHOR_CONCEPTS]
    # A sourcer writes an anchor of four to six terms, not seventeen. The cap
    # keeps the reference under the reward's own padding gate, and keeps the
    # eval's ceiling a thing a person would actually type.
    terms = [term for concept in picked for term in _usable_forms(concept)]
    return (
        finish(render_group(terms[:MAX_ANCHOR_TERMS]), lexicon),
        tuple(c.id for c in picked),
    )


# ---------------------------------------------------------------------------
# Task 1 from the lexicon
# ---------------------------------------------------------------------------


def register_prompts_from_lexicon(
    lexicon: Lexicon,
    ids: Sequence[str],
    split_name: str,
    *,
    skip: set[str] = frozenset(),
    reference_lexicon: Lexicon | None = None,
) -> list[Prompt]:
    """Two prompts per form-rich concept: a bare term, and a thin group.

    The bare case is the one Cloris's own corpus fails on 340 times over; the
    thin case is the one the enricher handles for listed concepts, and is here
    so the policy learns to finish a group rather than only to start one.

    ``skip`` carries the umbrella and generic concepts. Asking for the full
    register of "LLM" is asking for a group the scorer charges as a wasted
    gate and the enricher refuses to write — measured, the enricher improves
    27 of 27 non-umbrella listed concepts and 0 of 21 umbrella ones, because
    it is forbidden to add those words. A prompt whose best answer is a
    penalised group teaches the penalised behaviour.
    """
    wanted = set(ids)
    out: list[Prompt] = []
    for concept in lexicon.concepts:
        if concept.id not in wanted or concept.id in skip:
            continue
        forms = _usable_forms(concept)
        if len(forms) < MIN_FORMS_FOR_REGISTER:
            continue
        reference = finish(render_group(forms), reference_lexicon or lexicon)
        out.append(
            Prompt(
                id=f"lex-bare-{concept.id}",
                task=REGISTER,
                split=split_name,
                source="lexicon",
                given=f'"{forms[0]}"',
                targets=(concept.id,),
                reference=reference,
                families=concept.families,
                meta={"shape": "bare", "expected_forms": len(forms)},
            )
        )
        if len(forms) >= 3:
            out.append(
                Prompt(
                    id=f"lex-thin-{concept.id}",
                    task=REGISTER,
                    split=split_name,
                    source="lexicon",
                    given=render_group(forms[:2]),
                    targets=(concept.id,),
                    reference=reference,
                    families=concept.families,
                    meta={"shape": "thin", "expected_forms": len(forms)},
                )
            )
    return out


# ---------------------------------------------------------------------------
# Task 2 from the lexicon
# ---------------------------------------------------------------------------


def anchor_prompts_from_lexicon(
    lexicon: Lexicon,
    ids: Sequence[str],
    split_name: str,
    *,
    reference_lexicon: Lexicon | None = None,
    per_family: int = 14,
    seed: int = SEED,
) -> list[Prompt]:
    """Synthesised sweeps: k named things of one family, anchor withheld.

    Real anchorless sweeps exist in the corpus but almost none of them is made
    entirely of held-out concepts, so the eval half has to be built. Building
    it from the lexicon is not a compromise: a sweep of three benchmark names
    the reward has never read is exactly the generalisation being claimed.
    """
    wanted = set(ids)
    reference_lexicon = reference_lexicon or lexicon
    by_family: dict[str, list[Concept]] = {}
    for concept in lexicon.concepts:
        if concept.id not in wanted or concept.names_a_class:
            continue
        for family in concept.families:
            by_family.setdefault(family, []).append(concept)

    out: list[Prompt] = []
    for family in sorted(by_family):
        members = sorted(by_family[family], key=lambda c: c.id)
        if len(members) < MIN_SWEEP_TERMS:
            continue
        reference, anchor_ids = anchor_reference(reference_lexicon, family)
        if not reference:
            continue
        rng = random.Random(f"{seed}:anchor:{family}")
        for index in range(per_family):
            size = min(len(members), MIN_SWEEP_TERMS + (index % (MAX_SWEEP_TERMS - 1)))
            picked = rng.sample(members, size)
            sweep = render_group(
                _usable_forms(concept)[0] for concept in sorted(picked, key=lambda c: c.id)
            )
            out.append(
                Prompt(
                    id=f"lex-sweep-{split_name}-{family}-{index:02d}",
                    task=ANCHOR,
                    split=split_name,
                    source="lexicon",
                    given=sweep,
                    targets=tuple(sorted(concept.id for concept in picked)),
                    reference=reference,
                    families=(family,),
                    meta={"sweep_size": size, "anchor_concepts": list(anchor_ids)},
                )
            )
    # Identical sweeps can fall out of two draws; the id is the identity.
    unique: dict[str, Prompt] = {}
    for prompt in out:
        unique.setdefault(prompt.given, prompt)
    return sorted(unique.values(), key=lambda p: p.id)


# ---------------------------------------------------------------------------
# Both tasks from the real corpora
# ---------------------------------------------------------------------------


def _families_of(lexicon: Lexicon, concept_id: str) -> tuple[str, ...]:
    for concept in lexicon.concepts:
        if concept.id == concept_id:
            return concept.families
    return ()


def load_corpus(name: str) -> list[dict[str, Any]]:
    path = split_mod.FIXTURE_DIR / f"{name}.json"
    return json.loads(path.read_text())["strings"]


def _side_for(concept_ids: set[str], a_ids: set[str], b_ids: set[str]) -> str | None:
    """Which split a prompt belongs to, or ``None`` when it straddles.

    A group carrying one A concept and one B concept cannot be a training
    prompt without paying reward on a held-out concept, and cannot be an eval
    prompt without measuring one the policy was trained on. It is dropped.
    """
    if not concept_ids:
        return None
    if concept_ids <= a_ids:
        return TRAIN
    if concept_ids <= b_ids:
        return EVAL
    return None


def prompts_from_corpus(
    lexicon: Lexicon,
    a_ids: set[str],
    b_ids: set[str],
    *,
    corpora: Sequence[str] = ("ml_runs_76", "real_plans"),
    skip: set[str] = frozenset(),
    reference_for=None,
) -> list[Prompt]:
    reference_for = reference_for or (lambda _side: lexicon)
    out: list[Prompt] = []
    seen_given: set[tuple[str, str]] = set()
    for corpus in corpora:
        for row in load_corpus(corpus):
            boolean = row["boolean"]
            parsed = parse(boolean)
            if not parsed.legal:
                continue
            groups = _groups_for(parsed.tree, lexicon)
            positive = [group for group in groups if not group.negated]
            shape = shape_of(groups, lexicon, anchored_by_filter=False)

            for index, group in enumerate(positive):
                mapped = set(group.mapped_ids)
                side = _side_for(mapped, a_ids, b_ids)
                if side is None or len(group.terms) < 2 or mapped & skip:
                    continue

                # Task 1: strip the group to one surface per concept and ask
                # for the group back. The inverse of the edit operator
                # `drop_forms_to_one_per_concept`, which is how the pair
                # fixture manufactures a worse string.
                kept: list[str] = []
                used: set[str] = set()
                for term, concept_id in zip(group.terms, group.concept_ids):
                    if concept_id is None or concept_id in used:
                        continue
                    used.add(concept_id)
                    kept.append(term.text)
                given = render_group(kept)
                reference = finish(
                    render_group(term.text for term in group.terms),
                    reference_for(side),
                )
                if given != reference and (REGISTER, given) not in seen_given:
                    seen_given.add((REGISTER, given))
                    out.append(
                        Prompt(
                            id=f"{corpus}-{row['id']}-g{index}",
                            task=REGISTER,
                            split=side,
                            source=corpus,
                            given=given,
                            targets=tuple(sorted(mapped)),
                            reference=reference,
                            families=tuple(
                                sorted(
                                    {
                                        family
                                        for concept_id in mapped
                                        for family in lexicon.family_reach(concept_id)
                                    }
                                )
                            ),
                            meta={"shape": "stripped", "base": row["id"]},
                        )
                    )

            # Task 2: a string the scorer calls anchorless, whose sweep group
            # is read entirely on one side of the split.
            if shape not in ("lone_sweep", "two_groups_no_anchor"):
                continue
            for index, group in enumerate(positive):
                mapped = set(group.mapped_ids)
                side = _side_for(mapped, a_ids, b_ids)
                if side is None or len(mapped) < 2:
                    continue
                if any(
                    concept.names_a_class
                    for concept in lexicon.concepts
                    if concept.id in mapped
                ):
                    continue
                given = render_group(term.text for term in group.terms)
                if (ANCHOR, given) in seen_given:
                    continue
                shared = set.intersection(
                    *(set(_families_of(lexicon, cid)) for cid in mapped)
                )
                anchored = [
                    (family, *anchor_reference(reference_for(side), family))
                    for family in sorted(shared)
                ]
                anchored = [item for item in anchored if item[1]]
                if not anchored:
                    continue
                family, reference, anchor_ids = anchored[0]
                seen_given.add((ANCHOR, given))
                out.append(
                    Prompt(
                        id=f"{corpus}-{row['id']}-s{index}",
                        task=ANCHOR,
                        split=side,
                        source=corpus,
                        given=given,
                        targets=tuple(sorted(mapped)),
                        reference=reference,
                        families=(family,),
                        meta={
                            "shape": shape,
                            "base": row["id"],
                            "anchor_concepts": list(anchor_ids),
                        },
                    )
                )
    return out


# ---------------------------------------------------------------------------
# The whole set
# ---------------------------------------------------------------------------


HEADROOM_BAND = 0.01


def has_headroom(prompt: Prompt) -> bool:
    """True when the reference actually beats the input, in reward terms.

    Fourteen of forty held-out register prompts turned out to have none: the
    concept's whole COMMON register was already in the given group and every
    form the reference added was ``occasional``, which the scorer credits at
    0.05 apiece into a value it then clamps at 1.0. The reference and the
    untouched input scored identically to four decimals.

    Such a prompt is worse than useless in GRPO. Advantages are normalised
    within the group of G completions of one prompt, so a prompt on which
    every legal answer scores the same contributes pure noise to the
    gradient, and on the eval side it inflates the tie count and hides real
    movement. The filter is applied last, after the reward exists, because
    only the reward can say whether a prompt has anywhere to go.
    """
    from ..reward import tasks as reward_tasks  # late: reward imports this module

    given = prompt.given if prompt.given.startswith("(") else f"({prompt.given})"
    floor = reward_tasks.reward_floor_for(prompt)
    best = reward_tasks.reward(prompt, prompt.reference)
    start = reward_tasks.reward(prompt, given) if prompt.task == REGISTER else floor
    return best > start + HEADROOM_BAND


def build_prompts(
    document: dict[str, Any] | None = None,
    plan: split_mod.SplitPlan | None = None,
) -> list[Prompt]:
    document = document or split_mod.load_full_lexicon_document()
    plan = plan or split_mod.load_split()
    full = build_lexicon(document)
    a_ids, b_ids = set(plan.a_ids), set(plan.b_ids)

    # Umbrella and generic concepts are penalty vocabulary. They stay in both
    # lexicons so the scorer can charge them, and they are never a register
    # target, because the best answer to "expand LLM" is a group the scorer
    # is built to refuse.
    skip = split_mod.protected_ids(document)

    # A training anchor is paid by the A lexicon, so its reference has to be
    # an anchor the A lexicon can read. Building it from the whole lexicon
    # produced references made of held-out class concepts, which the reward
    # then floored — the ceiling sitting below the floor.
    reference_lexicons = {
        TRAIN: build_lexicon(split_mod.subset_document(document, plan.a_ids)),
        EVAL: full,
    }

    out: list[Prompt] = []
    out += register_prompts_from_lexicon(
        full, plan.a_ids, TRAIN, skip=skip, reference_lexicon=reference_lexicons[TRAIN]
    )
    out += register_prompts_from_lexicon(
        full, plan.b_ids, EVAL, skip=skip, reference_lexicon=reference_lexicons[EVAL]
    )
    out += anchor_prompts_from_lexicon(
        full, plan.a_ids, TRAIN, reference_lexicon=reference_lexicons[TRAIN]
    )
    out += anchor_prompts_from_lexicon(
        full, plan.b_ids, EVAL, reference_lexicon=reference_lexicons[EVAL]
    )
    out += prompts_from_corpus(
        full, a_ids, b_ids, skip=skip, reference_for=reference_lexicons.__getitem__
    )

    out = [prompt for prompt in out if has_headroom(prompt)]

    # The guarantee, re-checked on the finished set rather than trusted to the
    # three builders that were each supposed to hold it.
    seen_ids: set[str] = set()
    for prompt in out:
        if prompt.id in seen_ids:
            raise AssertionError(f"two prompts share the id {prompt.id!r}")
        seen_ids.add(prompt.id)
        targets = set(prompt.targets)
        expected = a_ids if prompt.split == TRAIN else b_ids
        if not targets <= expected:
            raise AssertionError(
                f"prompt {prompt.id} is a {prompt.split} prompt targeting "
                f"{sorted(targets - expected)}"
            )
    return sorted(out, key=lambda p: (p.split, p.task, p.id))


def counts(prompts: Sequence[Prompt]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for prompt in prompts:
        tally[f"{prompt.split}/{prompt.task}"] = (
            tally.get(f"{prompt.split}/{prompt.task}", 0) + 1
        )
        key = f"{prompt.split}/{prompt.task}/{prompt.source}"
        tally[key] = tally.get(key, 0) + 1
    return dict(sorted(tally.items()))


def write_prompts(prompts: Sequence[Prompt], path: str | Path | None = None) -> Path:
    target = Path(path or PROMPT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "role": split_mod.ROLE,
                "seed": SEED,
                "counts": counts(prompts),
                "prompts": [prompt.to_dict() for prompt in prompts],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return target


def load_prompts(path: str | Path | None = None) -> list[Prompt]:
    saved = json.loads(Path(path or PROMPT_PATH).read_text())
    return [prompt_from_dict(row) for row in saved["prompts"]]


def select(
    prompts: Sequence[Prompt],
    *,
    split: str | None = None,
    task: str | None = None,
    limit: int | None = None,
    seed: int = SEED,
    balance: bool = True,
) -> list[Prompt]:
    """A deterministic subset, balanced across tasks unless told otherwise."""
    pool = [
        prompt
        for prompt in prompts
        if (split is None or prompt.split == split)
        and (task is None or prompt.task == task)
    ]
    pool.sort(key=lambda p: p.id)
    if limit is None or len(pool) <= limit:
        return pool
    if not balance or task is not None:
        rng = random.Random(seed)
        return sorted(rng.sample(pool, limit), key=lambda p: p.id)
    picked: list[Prompt] = []
    tasks = sorted({prompt.task for prompt in pool})
    per_task = max(1, limit // len(tasks))
    for name in tasks:
        subset = [prompt for prompt in pool if prompt.task == name]
        rng = random.Random(f"{seed}:{name}")
        picked += (
            sorted(rng.sample(subset, per_task), key=lambda p: p.id)
            if len(subset) > per_task
            else subset
        )
    return sorted(picked, key=lambda p: p.id)[:limit]
