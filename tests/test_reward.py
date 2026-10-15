"""The reward: pure, deterministic, and unbribable.

Every test here is a hack the reward has to refuse. A reward that can be
gamed does not train a policy, it trains a gamer, and the cheapest hacks on a
coverage-shaped reward are all the same move — say more words.
"""

from __future__ import annotations

import math

import pytest

from cloris_boolean_rl.data import prompts as prompts_mod
from cloris_boolean_rl.data import split as split_mod
from cloris_boolean_rl.reward import lexicons, tasks
from cloris_boolean_rl.scorer.enrich import enrich
from cloris_boolean_rl.scorer.score import score

# The most a legal group about the wrong concept can earn on a register
# prompt: a quarter of the way from the floor to a perfect log composite.
OFF_TARGET_CEILING = tasks.FLOOR + 0.25 * (0.0 - tasks.FLOOR)


@pytest.fixture(scope="module")
def all_prompts():
    return prompts_mod.load_prompts()


@pytest.fixture(scope="module")
def train_register(all_prompts):
    return [
        p
        for p in all_prompts
        if p.split == "train" and p.task == "register" and p.source == "lexicon"
    ]


@pytest.fixture(scope="module")
def eval_register(all_prompts):
    return [
        p
        for p in all_prompts
        if p.split == "eval" and p.task == "register" and p.source == "lexicon"
    ]


@pytest.fixture(scope="module")
def eval_anchor(all_prompts):
    return [p for p in all_prompts if p.split == "eval" and p.task == "anchor"]


@pytest.fixture(scope="module")
def train_anchor(all_prompts):
    return [p for p in all_prompts if p.split == "train" and p.task == "anchor"]


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_reward_is_deterministic(train_register):
    prompt = train_register[0]
    first = tasks.reward(prompt, prompt.reference)
    second = tasks.reward(prompt, prompt.reference)
    assert first == second


def test_reward_never_mutates_its_inputs(train_register):
    prompt = train_register[0]
    before = prompt.to_dict()
    tasks.reward(prompt, prompt.reference)
    assert prompt.to_dict() == before


# ---------------------------------------------------------------------------
# The legality gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "   ",
        "()",
        '("eval"',
        '"eval" OR "evals"',  # no parentheses: not a group
        '("eval") AND ("RLHF")',  # two groups
        '("eval") NOT ("intern")',  # a NOT is not asked for
        'Here you go: ("eval" OR "evals")',  # prose
        '("eval" OR "evals") ("RLHF")',  # implicit AND
        '("eval" OR eval*)',  # wildcard; inside quotes a * is a literal
    ],
)
def test_illegal_or_multi_group_completions_floor(train_register, completion):
    prompt = train_register[0]
    assert tasks.reward(prompt, completion) == tasks.FLOOR


def test_an_empty_group_floors(train_register):
    assert tasks.reward(train_register[0], "()") == tasks.FLOOR


# ---------------------------------------------------------------------------
# The hacks
# ---------------------------------------------------------------------------


def test_a_lexicon_dump_floors(train_register):
    """The cheapest hack on a coverage reward: OR the whole word list.

    The scorer alone does NOT refuse this — measured, a sixty-term dump of A
    surfaces composites at 0.51, above a bare term's 0.41 — because parsimony
    charges no group-length term. The focus gate is what refuses it.
    """
    prompt = train_register[0]
    lexicon = lexicons.gate_lexicon("train")
    surfaces = [
        form.text
        for concept in lexicon.concepts
        for form in concept.forms
        if form.register == "common"
    ][:60]
    dump = prompts_mod.render_group(surfaces)
    assert tasks.reward(prompt, dump) == tasks.FLOOR


def test_an_umbrella_group_earns_no_more_than_off_target_credit(train_register):
    """An umbrella group names a concept outside the target's family, so it
    is graded as off-concept: above prose, never near a real answer."""
    prompt = train_register[0]
    umbrella = '("AI" OR "machine learning" OR "artificial intelligence" OR "deep learning")'
    outcome = tasks.explain(prompt, umbrella)
    assert outcome.gate == "off_concept"
    assert tasks.FLOOR < outcome.value <= OFF_TARGET_CEILING


def test_a_generic_verb_group_scores_far_below_the_reference(train_register):
    prompt = next(p for p in train_register if p.meta.get("expected_forms", 0) >= 4)
    verbs = '("led" OR "managed" OR "built" OR "shipped")'
    assert tasks.reward(prompt, verbs) < tasks.reward(prompt, prompt.reference) - 1.0


def test_off_concept_padding_scores_far_below_the_bare_target(train_register):
    """Answering a register prompt with a different family's vocabulary.

    Three concepts, five terms: under the focus gate's ceilings, so this is
    graded as off-concept rather than floored as padded. What must hold is
    that bolting another family's names onto the target is worth far less
    than the target alone.
    """
    prompt = next(p for p in train_register if p.targets[0] == "evaluation")
    padded = '("evaluation" OR "Pinecone" OR "Weaviate" OR "Milvus" OR "Qdrant")'
    outcome = tasks.explain(prompt, padded)
    assert outcome.gate == "off_concept"
    assert tasks.FLOOR < outcome.value <= OFF_TARGET_CEILING
    assert outcome.value < tasks.reward(prompt, '("evaluation")') - 5.0


# ---------------------------------------------------------------------------
# Graded misses: legal but about the wrong thing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evaluation_prompt(train_register):
    return next(p for p in train_register if p.targets[0] == "evaluation")


ON_TARGET = '("evaluation" OR "eval" OR "evals")'


def test_the_off_target_credit_is_a_quarter():
    assert tasks.OFF_TARGET_CREDIT == 0.25


def test_a_sibling_concept_is_off_target_and_graded(evaluation_prompt):
    """Same family as the target, but not the target: legal, wrong, and
    worth more than prose so GRPO can tell the two apart."""
    sibling = '("LLM-as-a-judge" OR "LLM judge")'
    outcome = tasks.explain(evaluation_prompt, sibling)
    assert outcome.gate == "off_target"
    assert tasks.FLOOR < outcome.value <= OFF_TARGET_CEILING
    assert outcome.value < tasks.reward(evaluation_prompt, ON_TARGET)
    assert outcome.group == sibling
    assert 0.0 < outcome.composite <= 1.0


def test_ignoring_the_prompt_is_graded_below_answering_it(evaluation_prompt):
    """Another family's concept entirely: off-concept, graded, far below."""
    elsewhere = '("RLHF" OR "DPO" OR "PPO")'
    outcome = tasks.explain(evaluation_prompt, elsewhere)
    assert outcome.gate == "off_concept"
    assert tasks.FLOOR < outcome.value <= OFF_TARGET_CEILING
    assert outcome.value < tasks.reward(evaluation_prompt, ON_TARGET)


def test_the_target_plus_an_unrelated_concept_is_graded(evaluation_prompt):
    mixed = '("evaluation" OR "RLHF")'
    outcome = tasks.explain(evaluation_prompt, mixed)
    assert outcome.gate == "off_concept"
    assert tasks.FLOOR < outcome.value <= OFF_TARGET_CEILING
    assert outcome.value < tasks.reward(evaluation_prompt, ON_TARGET)


def test_the_off_target_grade_follows_the_scorer(evaluation_prompt):
    """The grade is the scorer's own log composite, scaled a quarter of the
    way up from the floor: a better-formed wrong group earns a little more."""
    lexicon = lexicons.reward_lexicon("train", "register")
    sibling = '("LLM-as-a-judge" OR "LLM judge")'
    s = min(score(sibling, lexicon).log_composite, 0.0)
    expected = tasks.FLOOR + tasks.OFF_TARGET_CREDIT * (s - tasks.FLOOR)
    assert tasks.reward(evaluation_prompt, sibling) == pytest.approx(expected)


def test_padding_by_concept_count_still_floors(evaluation_prompt):
    """Six concepts, all in the target's family: over MAX_CONCEPTS, so the
    padded gate refuses it flat, with no graded credit."""
    six = (
        '("evaluation" OR "eval harness" OR "model evaluation" OR "LLM judge"'
        ' OR "autograder" OR "red teaming")'
    )
    outcome = tasks.explain(evaluation_prompt, six)
    assert outcome.gate == "padded"
    assert outcome.value == tasks.FLOOR


def test_padding_by_term_count_still_floors(evaluation_prompt):
    words = prompts_mod.render_group([f"term{i}" for i in range(tasks.MAX_TERMS + 1)])
    outcome = tasks.explain(evaluation_prompt, words)
    assert outcome.gate == "padded"
    assert outcome.value == tasks.FLOOR


def test_prose_still_floors(evaluation_prompt):
    outcome = tasks.explain(evaluation_prompt, "Evaluation is about measuring models.")
    assert outcome.gate == "contract"
    assert outcome.value == tasks.FLOOR


# ---------------------------------------------------------------------------
# The reward pays for the behaviour it is supposed to pay for
# ---------------------------------------------------------------------------


def test_the_enricher_never_lowers_the_reward_on_a_listed_concept(train_register):
    lexicon = lexicons.reward_lexicon("train")
    for prompt in train_register:
        bare = prompts_mod.render_group([prompt.given.strip('"')])
        enriched = enrich(bare, lexicon).enriched
        assert tasks.reward(prompt, enriched) >= tasks.reward(prompt, bare) - 1e-9


def test_the_enricher_lifts_every_concept_whose_register_it_can_read(train_register):
    """Where the enricher has forms to add, it must add them and score better.

    Where it has none — a concept whose extra spellings the lexicon marks
    ``occasional``, which the enricher is forbidden to inject — it ties, and
    that tie is the headroom the policy is being trained to take.
    """
    lexicon = lexicons.reward_lexicon("train")
    by_id = {concept.id: concept for concept in lexicon.concepts}
    lifted = headroom = 0
    for prompt in train_register:
        concept = by_id[prompt.targets[0]]
        common = [form for form in concept.forms if form.register == "common"]
        bare = prompts_mod.render_group([prompt.given.strip('"')])
        enriched = enrich(bare, lexicon).enriched
        better = tasks.reward(prompt, enriched) > tasks.reward(prompt, bare) + 1e-9
        if len(common) >= 2 and prompt.meta.get("shape") == "bare":
            assert better, f"{prompt.id}: enricher added nothing it could have added"
            lifted += 1
        elif not better:
            headroom += 1
    assert lifted >= 15
    assert headroom >= 15, "expected prompts the enricher cannot reach at all"


def test_the_reference_is_near_the_top(train_register):
    prompt = next(p for p in train_register if p.meta.get("expected_forms", 0) >= 4)
    reference = tasks.reward(prompt, prompt.reference)
    bare = tasks.reward(prompt, prompts_mod.render_group([prompt.given.strip('"')]))
    assert reference > bare
    assert reference > -1.0, "a fully covered concept should sit near log(1)"


# ---------------------------------------------------------------------------
# The split guarantee
# ---------------------------------------------------------------------------


def test_the_training_reward_cannot_read_any_held_out_surface():
    document = split_mod.load_full_lexicon_document()
    plan = split_mod.load_split()
    lexicon = lexicons.reward_lexicon("train")
    b_only = split_mod.surfaces_for(document, plan.b_ids) - split_mod.surfaces_for(
        document, plan.a_ids
    )
    assert b_only
    assert [term for term in b_only if lexicon.concept_for(term) is not None] == []


def test_the_eval_register_reward_reads_only_held_out_concepts():
    plan = split_mod.load_split()
    lexicon = lexicons.reward_lexicon("eval", task="register")
    assert {concept.id for concept in lexicon.concepts} == set(plan.b_ids)


def test_a_training_answer_earns_only_off_target_credit_on_an_eval_prompt(eval_register):
    """Memorised A vocabulary must not pay on a held-out concept: it is
    graded as off-concept, never passes as an answer, and sits far below
    even the untouched held-out input."""
    prompt = eval_register[0]
    a_dump = '("RLHF" OR "DPO" OR "PPO" OR "SFT")'
    outcome = tasks.explain(prompt, a_dump)
    assert outcome.gate == "off_concept"
    assert outcome.value <= OFF_TARGET_CEILING
    bare = prompts_mod.render_group([prompt.given.strip('"')])
    assert outcome.value < tasks.reward(prompt, bare) - 5.0


def test_eval_references_beat_their_bare_inputs(eval_register):
    better = 0
    for prompt in eval_register:
        bare = prompts_mod.render_group([prompt.given.strip('"')])
        if tasks.reward(prompt, prompt.reference) > tasks.reward(prompt, bare) + 1e-9:
            better += 1
    assert better >= 0.8 * len(eval_register)


# ---------------------------------------------------------------------------
# Task 2
# ---------------------------------------------------------------------------


def test_the_reference_anchor_sits_far_above_the_floor(train_anchor, eval_anchor):
    """What GRPO needs is the ORDERING, not a positive number.

    The anchor reward is a delta, and a correct anchor whose family the
    lexicon calls only ``adjacent`` can net slightly negative: relevance is
    the narrowest gate's, so pinning a core sweep with an adjacent class costs
    more than the shape gains. Measured, the reference delta runs from -0.04
    to +0.92 with a median near +0.6, and every one of them sits thirteen log
    units above what a refused proposal is worth. That gap is the gradient.
    """
    deltas = []
    for prompt in list(train_anchor) + list(eval_anchor):
        value = tasks.reward(prompt, prompt.reference)
        floor = tasks.reward_floor_for(prompt)
        assert value > floor + 5.0, f"{prompt.id}: reference scored {value:.3f}"
        deltas.append(value)
    deltas.sort()
    assert deltas[len(deltas) // 2] > 0.25


def test_a_sweep_extension_is_not_an_anchor(eval_anchor):
    """Answering 'name the class' with more named things must floor."""
    prompt = eval_anchor[0]
    more_names = '("SWE-bench" OR "HumanEval" OR "LiveCodeBench")'
    assert tasks.reward(prompt, more_names) == tasks.reward_floor_for(prompt)


def test_an_umbrella_only_anchor_floors(eval_anchor):
    prompt = eval_anchor[0]
    assert tasks.reward(prompt, '("AI" OR "machine learning")') == tasks.reward_floor_for(
        prompt
    )


def test_an_unrelated_anchor_floors(eval_anchor):
    prompt = next(p for p in eval_anchor if "benchmark" in p.families[0])
    unrelated = '("vector search" OR "vector database" OR "embeddings")'
    assert tasks.reward(prompt, unrelated) == tasks.reward_floor_for(prompt)


def test_the_anchor_reward_is_a_delta_against_the_sweep(train_anchor):
    prompt = train_anchor[0]
    outcome = tasks.explain(prompt, prompt.reference)
    assert outcome.gate == "ok"
    assert outcome.value == pytest.approx(outcome.combined - outcome.baseline)


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def test_the_reward_is_fast_enough_for_grpo(all_prompts):
    import time

    sample = all_prompts[:64]
    start = time.perf_counter()
    for prompt in sample:
        tasks.reward(prompt, prompt.reference)
    elapsed = time.perf_counter() - start
    # G=8 over 300 prompts is 2,400 calls per epoch; 5 ms each is 12 seconds.
    assert elapsed / len(sample) < 0.005


def test_floor_matches_the_scorers_own_zero():
    assert tasks.FLOOR == math.log(1e-6)
