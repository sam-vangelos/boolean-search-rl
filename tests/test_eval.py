"""The eval: the baselines are honest, and the metrics say what they mean."""

from __future__ import annotations

import pytest

from cloris_boolean_rl.data.prompts import ANCHOR, EVAL, REGISTER, load_prompts, select
from cloris_boolean_rl.eval import baselines, harness, metrics
from cloris_boolean_rl.reward import tasks


@pytest.fixture(scope="module")
def held_out():
    return select(load_prompts(), split=EVAL)


@pytest.fixture(scope="module")
def deterministic(held_out):
    return harness.run_deterministic(held_out)


def test_the_bare_baseline_is_the_input_unchanged(held_out):
    for prompt in held_out:
        if prompt.task == REGISTER:
            answer = baselines.bare_answer(prompt)
            assert tasks.extract_group(answer) is not None
            assert set(tasks._terms_of(answer)) == set(tasks._terms_of(prompt.given))
        else:
            assert baselines.bare_answer(prompt) == baselines.NO_PROPOSAL


def test_the_enricher_baseline_never_invents_a_term(held_out):
    """It may only add what the A lexicon already declares, or a hyphen twin.

    On held-out concepts that leaves it nothing to add, which is the point.
    """
    for prompt in held_out:
        if prompt.task != REGISTER:
            continue
        before = {t.normalized for t in tasks._terms_of(baselines.bare_answer(prompt))}
        after = {t.normalized for t in tasks._terms_of(baselines.enricher_answer(prompt))}
        assert before <= after


def test_the_enricher_cannot_help_on_held_out_concepts(held_out, deterministic):
    """The measurement Stage 3 exists to close, asserted so it cannot rot."""
    bare = deterministic[baselines.BARE].primary
    enriched = deterministic[baselines.ENRICHER].primary
    assert bare == enriched


def test_no_baseline_can_answer_an_anchor_prompt(held_out):
    for prompt in held_out:
        if prompt.task != ANCHOR:
            continue
        floor = tasks.reward_floor_for(prompt)
        assert tasks.reward(prompt, baselines.bare_answer(prompt)) == floor
        assert tasks.reward(prompt, baselines.enricher_answer(prompt)) == floor


def test_the_reference_ceiling_beats_both_baselines_everywhere(deterministic):
    reference = deterministic[baselines.REFERENCE].primary
    for name in (baselines.BARE, baselines.ENRICHER):
        rates = metrics.win_rate(reference, deterministic[name].primary)
        assert rates["losses"] == 0
        assert rates["win_rate"] == 1.0


def test_win_rate_counts_ties_rather_than_splitting_them():
    rates = metrics.win_rate([1.0, 0.0, 0.5], [0.0, 1.0, 0.5])
    assert rates == {
        "n": 3,
        "wins": 1,
        "losses": 1,
        "ties": 1,
        "win_rate": round(1 / 3, 4),
        "decisive_win_rate": 0.5,
    }


def test_diversity_sees_collapse():
    same = ['("eval" OR "evals")'] * 8
    collapsed = metrics.diversity(same)
    assert collapsed.distinct == 1
    assert collapsed.mean_pairwise_jaccard == 1.0

    varied = metrics.diversity(
        [
            '("eval")',
            '("evals")',
            '("evaluation" OR "evaluations")',
            '("eval" OR "evals" OR "evaluation")',
        ]
    )
    assert varied.distinct == 4
    assert varied.mean_pairwise_jaccard < 0.5


def test_diversity_counts_illegal_samples():
    result = metrics.diversity(['("eval")', "sorry, I cannot", ""])
    assert result.illegal == 2


def test_the_report_carries_everything_a_reader_needs(held_out, deterministic):
    report = harness.build_report(held_out, deterministic)
    assert report["prompts"]["n"] == len(held_out)
    assert set(report["systems"]) == set(baselines.DETERMINISTIC)
    for entry in report["systems"].values():
        assert "best_of_n" in entry and "gates" in entry and "by_task" in entry
    assert "reference_vs_bare" in report["win_rates"]
    assert harness.render(report).splitlines()[0].startswith("held-out prompts:")


def test_the_gate_tally_names_why_a_baseline_failed(held_out, deterministic):
    gates = harness.build_report(held_out, deterministic)["systems"][baselines.BARE][
        "gates"
    ]
    assert gates.get("contract", 0) >= 1, "anchor prompts have no bare answer"
    assert gates.get("ok", 0) >= 1


# ---------------------------------------------------------------------------
# Per-prompt detail: which prompts tied, and why; and the typical answer
# ---------------------------------------------------------------------------

PROSE = "sorry, I cannot help with that"
G = 8


@pytest.fixture(scope="module")
def three(held_out):
    """Three register prompts: one to tie at the ceiling, one to tie on the
    floor, one to tie in between."""
    return [p for p in held_out if p.task == REGISTER][:3]


def _sampled(name, prompts, table):
    """A stub generator: ``table`` maps a prompt id to its G completions."""

    def generate(batch, samples, temperature):
        return [list(table[p.id])[:samples] for p in batch]

    return harness.run_sampled(name, prompts, generate, samples=G)


def _systems(prompts, base_table, checkpoint_table):
    systems = harness.run_deterministic(prompts)
    systems[baselines.BASE] = _sampled(baselines.BASE, prompts, base_table)
    systems["checkpoint"] = _sampled("checkpoint", prompts, checkpoint_table)
    return systems


def _lucky_once(prompts):
    """Base finds the reference once in eight, checkpoint every time."""
    base = {p.id: [PROSE] * (G - 1) + [p.reference] for p in prompts}
    checkpoint = {p.id: [p.reference] * G for p in prompts}
    return base, checkpoint


def test_every_prompt_gets_a_per_prompt_entry(three):
    report = harness.build_report(three, _systems(three, *_lucky_once(three)))
    rows = report["per_prompt"]
    assert [row["id"] for row in rows] == [p.id for p in three]
    for prompt, row in zip(three, rows):
        assert row["task"] == prompt.task
        assert row["reference"] == pytest.approx(tasks.reward(prompt, prompt.reference), abs=1e-4)
        for name in baselines.DETERMINISTIC:
            assert len(row["systems"][name]["rewards"]) == 1
        for name in (baselines.BASE, "checkpoint"):
            cell = row["systems"][name]
            assert len(cell["rewards"]) == G
            assert cell["best"] == max(cell["rewards"])
            assert cell["best_gate"] == "ok"
        base = row["systems"][baselines.BASE]
        assert base["usable"] == 1 and base["at_reference"] == 1
        checkpoint = row["systems"]["checkpoint"]
        assert checkpoint["usable"] == G and checkpoint["at_reference"] == G


def test_the_typical_answer_separates_what_best_of_n_ties(three):
    report = harness.build_report(three, _systems(three, *_lucky_once(three)))
    best = report["win_rates"]["checkpoint_vs_base"]
    typical = report["win_rates_typical"]["checkpoint_vs_base"]
    assert best["ties"] == len(three) and best["wins"] == 0
    assert typical["wins"] == len(three) and typical["ties"] == 0
    assert set(report["win_rates_typical"]) == set(report["win_rates"])


def test_usable_and_at_reference_rates(three):
    report = harness.build_report(three, _systems(three, *_lucky_once(three)))
    base = report["systems"][baselines.BASE]
    checkpoint = report["systems"]["checkpoint"]
    assert base["usable_rate"] == round(1 / G, 4)
    assert base["at_reference_rate"] == round(1 / G, 4)
    assert checkpoint["usable_rate"] == 1.0
    assert checkpoint["at_reference_rate"] == 1.0
    assert "usable_rate" not in report["systems"][baselines.BARE]


def test_gates_best_reads_the_best_sample_not_the_first(three):
    report = harness.build_report(three, _systems(three, *_lucky_once(three)))
    base = report["systems"][baselines.BASE]
    assert base["gates"] == {"contract": len(three)}, "the old tally is left alone"
    assert base["gates_best"] == {"ok": len(three)}


def test_the_tie_breakdown_names_ceiling_floor_and_partial_ties(three):
    ceiling, floor, partial = three
    table = {
        ceiling.id: [PROSE] * (G - 1) + [ceiling.reference],
        floor.id: [PROSE] * G,
        partial.id: [PROSE] * (G - 1) + [baselines.bare_answer(partial)],
    }
    report = harness.build_report(three, _systems(three, table, dict(table)))
    assert report["win_rates"]["checkpoint_vs_base"]["ties"] == 3
    breakdown = report["tie_breakdown"]
    assert set(breakdown) == {"checkpoint_vs_base", "base_vs_checkpoint"} & set(
        report["win_rates"]
    )
    tie = breakdown["checkpoint_vs_base"]
    assert tie["ties"] == 3
    assert tie["buckets"]["both_at_reference"] == {"n": 1, "ids": [ceiling.id]}
    assert tie["buckets"]["both_failed"] == {"n": 1, "ids": [floor.id]}
    assert tie["buckets"]["both_partial"] == {"n": 1, "ids": [partial.id]}
    rendered = harness.render(report)
    assert "typical" in rendered and "both_at_reference" in rendered


def test_the_tie_helper_is_the_comparison_win_rate_uses():
    assert metrics.compare(1.0, 0.0) == "win"
    assert metrics.compare(0.0, 1.0) == "loss"
    assert metrics.compare(0.5, 0.5 + metrics.TIE_BAND / 2) == "tie"
