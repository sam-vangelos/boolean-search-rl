"""Which lexicon each reward reads, and why it is not always the same one.

Three lexicons, built once from the A/B split and cached, because GRPO calls
the reward G times per prompt per step and a re-parse of a 77 KB JSON file per
call would dominate the step time.

``reward_lexicon("train")`` is A. Every unit of training reward is paid by it,
and it cannot read a single held-out surface. That is the exact statement the
whole Stage 3 claim rests on, and ``test_split.py`` checks it as an identity
rather than as a convention.

``reward_lexicon("eval", task="register")`` is B and only B. A register prompt
asks for one held-out concept's spellings, so B alone can read the answer and
any memorised A vocabulary in the completion is simply invisible — unmapped,
uncredited, and then refused outright by the focus gate. This is the literal
form of "the eval reward never reads A-only data".

``reward_lexicon("eval", task="anchor")`` is the WHOLE lexicon, and this is a
deliberate departure. The held-out thing in an anchor prompt is the SWEEP: a
set of named artifacts the training reward never read. The answer is general
class vocabulary — "benchmark", "evaluation", "post-training" — which the
split may well have left in A, and a B-only lexicon would floor a correct
anchor for the crime of being correct in A's words. So the anchor reward reads
everything, and the generalisation claim is carried by the sweep side plus the
family gate: the proposed anchor must be the right class for a sweep the
policy has never been paid to recognise.

The GATE lexicon is separate again. A gate can only ever refuse a completion,
never credit one, so letting it read more than the reward does cannot teach
the policy a word it was not paid for. At eval it reads the whole lexicon,
which is what lets it recognise an A-vocabulary dump as a dump instead of as
harmless unmapped text. At training it reads A, so training touches nothing
else.
"""

from __future__ import annotations

from functools import lru_cache

from ..data import split as split_mod
from ..scorer.lexicon import Lexicon, build_lexicon

TRAIN = "train"
EVAL = "eval"
REGISTER = "register"
ANCHOR = "anchor"


@lru_cache(maxsize=1)
def full_lexicon() -> Lexicon:
    return build_lexicon(split_mod.load_full_lexicon_document())


@lru_cache(maxsize=2)
def _half(side: str) -> Lexicon:
    document = split_mod.load_full_lexicon_document()
    plan = split_mod.load_split()
    ids = plan.a_ids if side == "a" else plan.b_ids
    return build_lexicon(split_mod.subset_document(document, ids))


def a_lexicon() -> Lexicon:
    return _half("a")


def b_lexicon() -> Lexicon:
    return _half("b")


def reward_lexicon(split: str, task: str = REGISTER) -> Lexicon:
    """The lexicon whose readings become reward."""
    if split == TRAIN:
        return a_lexicon()
    if task == ANCHOR:
        return full_lexicon()
    return b_lexicon()


def gate_lexicon(split: str) -> Lexicon:
    """The lexicon the refusal gates read. It can only ever subtract."""
    return a_lexicon() if split == TRAIN else full_lexicon()
