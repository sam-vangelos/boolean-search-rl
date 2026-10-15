"""Split the role lexicon's concepts into A (rewarded) and B (held out).

The whole Stage 3 claim is that the policy learns a BEHAVIOUR — expand a term
into the forms people actually write, name the class a sweep belongs to — and
not a word list. A policy trained against a reward that can read every concept
in the lexicon would pass any eval built from that same lexicon by memorising
it, and the pass would mean nothing. So the concepts are cut in two before any
prompt is built: A pays the training reward and supplies every training
target, B is never rewarded and is the only thing the eval measures.

Three rules shape the cut.

Stratify by family, so the held-out set is not an accident of alphabetical
order. Each family surrenders roughly a quarter of its eligible concepts, and
a family holding two or more of them surrenders at least one; no family is
emptied, because the A-lexicon has to remain a usable lexicon.

Protect the umbrella and the generic. ``ai``, ``machine learning``, ``LLM``
and the ubiquitous verbs are penalty vocabulary: the scorer charges a group
made of them, and it can only charge what it can read. Holding one out would
quietly switch off a penalty rather than hold out a concept, so those stay in
A and stay in both sub-lexicons' term sets.

Cut whole concepts, never forms. Half a concept's register in A and half in B
is a leak wearing a split's clothes.

The result is written to ``data/splits/re-v1.json`` and read from there by
everything downstream, so the split is an artifact under review rather than a
function anyone can re-seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..scorer.lexicon import LEXICON_DIR, normalize

DATA_DIR = Path(__file__).resolve().parent
SPLIT_DIR = DATA_DIR / "splits"
SPLIT_PATH = SPLIT_DIR / "re-v1.json"
FIXTURE_DIR = DATA_DIR / "fixtures"

ROLE = "research-engineer-post-training"
LEXICON_PATH = LEXICON_DIR / f"{ROLE}.json"

SEED = 20260921
HOLDOUT_SHARE = 0.25


@dataclass(frozen=True)
class SplitPlan:
    """Which concept ids pay the reward and which are only ever measured."""

    role: str
    seed: int
    a_ids: tuple[str, ...]
    b_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "seed": self.seed,
            "holdout_share": HOLDOUT_SHARE,
            "counts": {"a": len(self.a_ids), "b": len(self.b_ids)},
            "a": list(self.a_ids),
            "b": list(self.b_ids),
        }


# ---------------------------------------------------------------------------
# Reading the source lexicon
# ---------------------------------------------------------------------------


def load_full_lexicon_document(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or LEXICON_PATH).read_text())


def families_of(concept: Mapping[str, Any]) -> tuple[str, ...]:
    declared = concept.get("families", concept.get("family"))
    if isinstance(declared, str):
        declared = [declared]
    return tuple(dict.fromkeys(str(item).strip() for item in declared or ()))


def primary_family(concept: Mapping[str, Any]) -> str:
    families = families_of(concept)
    return families[0] if families else "unclassified"


def surfaces_of(concept: Mapping[str, Any]) -> set[str]:
    return {normalize(form["text"]) for form in concept.get("forms", ())}


def surfaces_for(document: Mapping[str, Any], ids: Iterable[str]) -> set[str]:
    wanted = set(ids)
    out: set[str] = set()
    for concept in document["concepts"]:
        if concept["id"] in wanted:
            out |= surfaces_of(concept)
    return out


def concepts_by_family(document: Mapping[str, Any]) -> dict[str, list[str]]:
    """Every concept id, filed under every family it declares."""
    out: dict[str, list[str]] = {}
    for concept in document["concepts"]:
        for family in families_of(concept):
            out.setdefault(family, []).append(concept["id"])
    return out


def protected_ids(document: Mapping[str, Any]) -> set[str]:
    """Concepts that must stay in A because the scorer charges their words.

    A concept any of whose surfaces is an umbrella term, a generic verb, an
    employer name or a noisy token is penalty vocabulary first and reward
    vocabulary second. Holding it out would switch a penalty off in the
    training lexicon, which is a silent change to the reward, not a holdout.
    """
    charged = set()
    for key in ("umbrella_terms", "generic_verbs", "employer_terms", "noisy_terms"):
        charged |= {normalize(term) for term in document.get(key) or ()}
    return {
        concept["id"]
        for concept in document["concepts"]
        if surfaces_of(concept) & charged
    }


def eligible_by_family(document: Mapping[str, Any]) -> dict[str, list[str]]:
    """Holdout candidates, filed under their PRIMARY family only.

    A concept sitting in three families must be assigned to one bucket, or the
    per-family quotas double-count it and the share drifts.
    """
    protected = protected_ids(document)
    out: dict[str, list[str]] = {}
    for concept in document["concepts"]:
        if concept["id"] in protected:
            continue
        out.setdefault(primary_family(concept), []).append(concept["id"])
    return out


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------


def build_split(
    document: Mapping[str, Any] | None = None,
    *,
    seed: int = SEED,
    share: float = HOLDOUT_SHARE,
) -> SplitPlan:
    document = document or load_full_lexicon_document()
    declared = [concept["id"] for concept in document["concepts"]]
    by_family = eligible_by_family(document)
    every_family = concepts_by_family(document)

    held: set[str] = set()
    for family in sorted(by_family):
        candidates = sorted(by_family[family])
        if len(candidates) < 2:
            # One eligible concept is the family's only readable vocabulary
            # once the protected ones are set aside; holding it out empties
            # the family from A.
            continue
        rng = random.Random(f"{seed}:{family}")
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        quota = max(1, round(share * len(candidates)))
        quota = min(quota, len(candidates) - 1)
        held.update(shuffled[:quota])

    # No family may be emptied, counting every family a concept declares and
    # not just its primary one: a family whose members are all somebody else's
    # secondary family still has to be readable in A.
    for family in sorted(every_family):
        members = set(every_family[family])
        if members and members <= held:
            returned = sorted(members)[0]
            held.discard(returned)

    a_ids = tuple(cid for cid in declared if cid not in held)
    b_ids = tuple(cid for cid in declared if cid in held)
    return SplitPlan(role=str(document.get("role", ROLE)), seed=seed, a_ids=a_ids, b_ids=b_ids)


# ---------------------------------------------------------------------------
# Sub-lexicons
# ---------------------------------------------------------------------------


def subset_document(
    document: Mapping[str, Any], keep_ids: Iterable[str]
) -> dict[str, Any]:
    """The lexicon document restricted to ``keep_ids``, still loadable.

    The penalty vocabulary — umbrella, generic verbs, employers, noisy tokens —
    survives whole in both halves, because those lists are what the scorer
    charges rather than what it rewards, and a half that cannot charge them is
    a different reward. Everything keyed to a concept is pruned with it:
    morphology entries whose key no longer names a surface, families nobody
    declares any more, and the parents and relevances of those families. The
    loader refuses a document that dangles any of them, which is the check
    that this pruning is complete.
    """
    keep = set(keep_ids)
    concepts = [c for c in document["concepts"] if c["id"] in keep]
    if not concepts:
        raise ValueError("a lexicon subset with no concepts is not a lexicon")

    surviving_surfaces: set[str] = set()
    surviving_families: set[str] = set()
    for concept in concepts:
        surviving_surfaces |= surfaces_of(concept)
        surviving_families |= set(families_of(concept))

    morphology = {
        key: values
        for key, values in (document.get("morphology") or {}).items()
        if normalize(key) in surviving_surfaces
    }
    family_parents = {
        family: parent
        for family, parent in (document.get("family_parents") or {}).items()
        if family in surviving_families
    }
    family_relevance = {
        family: level
        for family, level in (document.get("family_relevance") or {}).items()
        if family in surviving_families
    }
    anchor_families = [
        family
        for family in (document.get("anchor_families") or [])
        if family in surviving_families
    ]

    out = dict(document)
    out["concepts"] = concepts
    out["morphology"] = morphology
    out["family_parents"] = family_parents
    out["family_relevance"] = family_relevance
    out["anchor_families"] = anchor_families
    return out


def load_split(path: str | Path | None = None) -> SplitPlan:
    saved = json.loads(Path(path or SPLIT_PATH).read_text())
    return SplitPlan(
        role=saved["role"],
        seed=int(saved["seed"]),
        a_ids=tuple(saved["a"]),
        b_ids=tuple(saved["b"]),
    )


def write_split(plan: SplitPlan, path: str | Path | None = None) -> Path:
    target = Path(path or SPLIT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    return target
