"""The three things a checkpoint has to beat, and the ceiling it aims at.

A trained policy that beats nothing is not worth an adapter file, so the
comparison is fixed before the first run rather than chosen after it.

BARE is the input unchanged. It is the floor of usefulness: the string Cloris
already has. On a register prompt it is the thin group; on an anchor prompt
there is nothing to return, because an anchorless sweep's "unchanged" answer
is no anchor at all, and that is the point of the task.

ENRICHER is ``linkedin/boolean_score/enrich.py`` reading the A lexicon — the
deterministic tool Cloris ships today, given exactly the vocabulary the
training reward had. On a held-out concept it can still add the derived
hyphen twin, because that axis is orthography rather than vocabulary, and
nothing else: it invents no forms by design. That is the gap Stage 3 exists
to close, and this baseline is its exact measurement. On an anchor prompt it
returns nothing, which is Stage 2's residue table restated as code —
``missing_anchor`` was the largest remaining defect on all three corpora
precisely because no amount of adding spellings creates an anchor.

BASE is the untrained policy, sampled eight times at the training temperature.
It is the only baseline that can answer both tasks, and the only one whose
loss to a checkpoint means the training did something.

REFERENCE is not a baseline. It is the best answer the lexicon knows, kept as
a ceiling so a win can be read as a fraction of what was available rather
than as a number floating on its own.
"""

from __future__ import annotations

from ..data.prompts import ANCHOR, REGISTER, Prompt
from ..reward import lexicons
from ..scorer.enrich import enrich

BARE = "bare"
ENRICHER = "enricher"
BASE = "base"
REFERENCE = "reference"

NO_PROPOSAL = ""


def bare_answer(prompt: Prompt) -> str:
    if prompt.task == REGISTER:
        text = prompt.given
        return text if text.startswith("(") else f"({text})"
    return NO_PROPOSAL


def enricher_answer(prompt: Prompt) -> str:
    """What Cloris's deterministic enricher would do, with A's vocabulary."""
    if prompt.task == ANCHOR:
        return NO_PROPOSAL
    lexicon = lexicons.a_lexicon()
    report = enrich(bare_answer(prompt), lexicon)
    return report.enriched if report.legal else bare_answer(prompt)


def reference_answer(prompt: Prompt) -> str:
    return prompt.reference


DETERMINISTIC = {
    BARE: bare_answer,
    ENRICHER: enricher_answer,
    REFERENCE: reference_answer,
}
