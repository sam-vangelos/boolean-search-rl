"""How a checkpoint is compared, and the one number that is watched not paid.

WIN RATE is pairwise and in log space, over the same prompt: a reward's job in
a training rig is to rank several completions of one prompt against each
other, so the eval asks the same question the reward does. The tie band is
0.01 log units — about a 1% relative difference — which is the band Cloris's
own preference fixture uses, so the two numbers mean the same thing.

DIVERSITY is measured and never rewarded, because it is the tell for the
failure mode GRPO has and supervised fine-tuning does not. A policy that has
collapsed emits the same group eight times: distinct-set count goes to 1 and
mean pairwise Jaccard goes to 1.0 while mean reward may look fine, because
every sample is the same locally-good answer. Rewarding diversity would fix
the symptom and break the reward — variety is not a virtue in a search
string, it is only evidence that the policy still has somewhere to go. So it
goes in the report and never in the reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Sequence

from ..reward import tasks

TIE_BAND = 0.01


def term_set(completion: str) -> frozenset[str]:
    group = tasks.extract_group(completion)
    if group is None:
        return frozenset()
    return frozenset(term.normalized for term in tasks._terms_of(group))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class Diversity:
    samples: int
    distinct: int
    mean_pairwise_jaccard: float
    illegal: int

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "distinct_term_sets": self.distinct,
            "mean_pairwise_jaccard": round(self.mean_pairwise_jaccard, 4),
            "illegal": self.illegal,
        }


def diversity(completions: Sequence[str]) -> Diversity:
    sets = [term_set(text) for text in completions]
    pairs = [jaccard(a, b) for a, b in combinations(sets, 2)]
    return Diversity(
        samples=len(completions),
        distinct=len({frozenset(item) for item in sets}),
        mean_pairwise_jaccard=mean(pairs) if pairs else 1.0,
        illegal=sum(1 for item in sets if not item),
    )


def compare(left: float, right: float, band: float = TIE_BAND) -> str:
    """One prompt's verdict, ``"win"``, ``"loss"`` or ``"tie"``, from the
    left side's view. The one place the tie band is applied, so a report that
    lists which prompts tied agrees with the win rate that counted them."""
    if left > right + band:
        return "win"
    if left < right - band:
        return "loss"
    return "tie"


def win_rate(
    challenger: Sequence[float], incumbent: Sequence[float], band: float = TIE_BAND
) -> dict[str, float]:
    """Pairwise over the same prompts. Ties are reported, never split."""
    wins = losses = ties = 0
    for left, right in zip(challenger, incumbent):
        outcome = compare(left, right, band)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            ties += 1
    total = wins + losses + ties
    return {
        "n": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "decisive_win_rate": (
            round(wins / (wins + losses), 4) if (wins + losses) else 0.0
        ),
    }


def describe(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "median": round(ordered[len(ordered) // 2], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def gate_tally(prompts, completions) -> dict[str, int]:
    """Why completions were refused, which is the difference between a
    formatting problem and a reward hack."""
    tally: dict[str, int] = {}
    for prompt, text in zip(prompts, completions):
        gate = gate_of(prompt, text)
        tally[gate] = tally.get(gate, 0) + 1
    return dict(sorted(tally.items(), key=lambda item: -item[1]))


def gate_of(prompt, completion: str) -> str:
    """The gate the task's reward applied: ``"ok"`` when it scored the group,
    otherwise the name of the refusal."""
    return tasks.explain(prompt, completion).gate
