"""Run every system over the held-out prompts and write one JSON report.

The deterministic half — bare, enricher, reference — needs no model and runs
in under a second on any machine, which is why it is the half that can be
measured before a GPU is ever rented. The sampled half needs a model and is
the same code path for the untrained base and for a checkpoint, because a
comparison between two things measured differently is not a comparison.

Every sampled system is drawn G times per prompt at the training temperature,
and reported three ways: MEAN over the samples, which is what GRPO's
expectation actually moves; BEST-OF-8, which is what the downstream retrieval
loop will consume, since it samples several strings and keeps the one that
retrieves; and DIVERSITY, which is watched for collapse and never paid.

The aggregates say how often a checkpoint won; the per-prompt rows say which
prompts, so a tie can be read without re-running a model. A tie where both
best answers already reach the reference is a ceiling and costs nothing; a
tie where neither produced a usable group is where the next run should look.
And because best-of-8 lets a weak policy draw level by getting lucky once,
every pairing is also reported on the typical (mean) answer.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

from ..data.prompts import EVAL, Prompt, load_prompts, select
from ..reward import tasks
from . import baselines, metrics

DEFAULT_SAMPLES = 8
DEFAULT_TEMPERATURE = 0.8


@dataclass
class SystemResult:
    name: str
    sampled: bool
    completions: list[list[str]] = field(default_factory=list)
    mean_rewards: list[float] = field(default_factory=list)
    best_rewards: list[float] = field(default_factory=list)
    sample_rewards: list[list[float]] = field(default_factory=list)
    sample_gates: list[list[str]] = field(default_factory=list)

    def best_index(self, prompt_index: int) -> int:
        """Which sample was the best for one prompt; the first, on a tie."""
        values = self.sample_rewards[prompt_index]
        return values.index(max(values))

    @property
    def primary(self) -> list[float]:
        """The series a win rate is computed on: best-of-G where there is a
        G, and the single answer where there is not."""
        return self.best_rewards


def score_system(
    name: str, prompts: Sequence[Prompt], completions: Sequence[Sequence[str]]
) -> SystemResult:
    result = SystemResult(name=name, sampled=len(completions[0]) > 1 if completions else False)
    for prompt, samples in zip(prompts, completions):
        outcomes = [tasks.explain(prompt, text) for text in samples]
        values = [outcome.value for outcome in outcomes]
        result.completions.append(list(samples))
        result.sample_rewards.append(values)
        result.sample_gates.append([outcome.gate for outcome in outcomes])
        result.mean_rewards.append(mean(values))
        result.best_rewards.append(max(values))
    return result


def run_deterministic(prompts: Sequence[Prompt]) -> dict[str, SystemResult]:
    out: dict[str, SystemResult] = {}
    for name, answer in baselines.DETERMINISTIC.items():
        out[name] = score_system(name, prompts, [[answer(p)] for p in prompts])
    return out


def run_sampled(
    name: str,
    prompts: Sequence[Prompt],
    generate: Callable[[Sequence[Prompt], int, float], list[list[str]]],
    *,
    samples: int = DEFAULT_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> SystemResult:
    return score_system(name, prompts, generate(prompts, samples, temperature))


def build_report(
    prompts: Sequence[Prompt],
    systems: dict[str, SystemResult],
    *,
    against: Sequence[str] = (baselines.BARE, baselines.ENRICHER, baselines.BASE),
    note: str = "",
) -> dict:
    tasks_present = sorted({p.task for p in prompts})
    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "note": note,
        "prompts": {
            "n": len(prompts),
            "by_task": {
                task: sum(1 for p in prompts if p.task == task) for task in tasks_present
            },
            "ids": [p.id for p in prompts],
        },
        "systems": {},
        "win_rates": {},
        "win_rates_typical": {},
        "tie_breakdown": {},
    }
    reference = systems.get(baselines.REFERENCE)
    ceiling = reference.best_rewards if reference is not None else None

    for name, result in systems.items():
        entry = {
            "sampled": result.sampled,
            "best_of_n": metrics.describe(result.best_rewards),
            "mean_of_n": metrics.describe(result.mean_rewards),
            "gates": metrics.gate_tally(prompts, [s[0] for s in result.completions]),
            "gates_best": metrics.gate_tally(
                prompts,
                [
                    samples[result.best_index(index)]
                    for index, samples in enumerate(result.completions)
                ],
            ),
            "by_task": {
                task: metrics.describe(
                    [
                        value
                        for prompt, value in zip(prompts, result.best_rewards)
                        if prompt.task == task
                    ]
                )
                for task in tasks_present
            },
        }
        if result.sampled:
            per_prompt = [metrics.diversity(samples) for samples in result.completions]
            entry["diversity"] = {
                "mean_distinct_term_sets": round(
                    mean(d.distinct for d in per_prompt), 3
                ),
                "mean_pairwise_jaccard": round(
                    mean(d.mean_pairwise_jaccard for d in per_prompt), 4
                ),
                "mode_collapsed_prompts": sum(1 for d in per_prompt if d.distinct == 1),
                "mean_illegal_per_prompt": round(mean(d.illegal for d in per_prompt), 3),
            }
            total = sum(len(values) for values in result.sample_rewards)
            entry["usable_rate"] = round(
                sum(gates.count("ok") for gates in result.sample_gates) / total, 4
            )
            if ceiling is not None:
                entry["at_reference_rate"] = round(
                    sum(
                        _at_reference(values, top)
                        for values, top in zip(result.sample_rewards, ceiling)
                    )
                    / total,
                    4,
                )
        report["systems"][name] = entry

    report["per_prompt"] = [
        {
            "id": prompt.id,
            "task": prompt.task,
            "reference": round(ceiling[index], 4) if ceiling is not None else None,
            "systems": {
                name: _prompt_cell(
                    result, index, ceiling[index] if ceiling is not None else None
                )
                for name, result in systems.items()
            },
        }
        for index, prompt in enumerate(prompts)
    ]

    for challenger in systems:
        for incumbent in against:
            if challenger == incumbent or incumbent not in systems:
                continue
            pairing = f"{challenger}_vs_{incumbent}"
            left, right = systems[challenger], systems[incumbent]
            report["win_rates"][pairing] = metrics.win_rate(left.primary, right.primary)
            report["win_rates_typical"][pairing] = metrics.win_rate(
                left.mean_rewards, right.mean_rewards
            )
            if left.sampled and right.sampled:
                report["tie_breakdown"][pairing] = _tie_breakdown(
                    prompts, left, right, ceiling
                )
    return report


def _at_reference(values: Sequence[float], top: float) -> int:
    """How many samples reach the reference: within the tie band of it, or
    above it, by the same comparison the win rate uses."""
    return sum(1 for value in values if metrics.compare(value, top) != "loss")


def _prompt_cell(result: SystemResult, index: int, top: float | None) -> dict:
    values = result.sample_rewards[index]
    gates = result.sample_gates[index]
    best = result.best_index(index)
    cell = {
        "rewards": [round(value, 4) for value in values],
        "best": round(values[best], 4),
        "mean": round(mean(values), 4),
        "best_gate": gates[best],
        "usable": gates.count("ok"),
    }
    if top is not None:
        cell["at_reference"] = _at_reference(values, top)
    return cell


TIE_BUCKETS = ("both_at_reference", "both_failed", "both_partial")


def _tie_breakdown(
    prompts: Sequence[Prompt],
    left: SystemResult,
    right: SystemResult,
    ceiling: Sequence[float] | None,
) -> dict:
    """Why each best-of-N tie tied. At the reference is a ceiling and fine;
    neither best sample passing its gate is a target for more training;
    anything else is a shared partial answer."""
    buckets: dict[str, list[str]] = {name: [] for name in TIE_BUCKETS}
    ties = 0
    for index, prompt in enumerate(prompts):
        a, b = left.primary[index], right.primary[index]
        if metrics.compare(a, b) != "tie":
            continue
        ties += 1
        if ceiling is not None and all(
            metrics.compare(value, ceiling[index]) != "loss" for value in (a, b)
        ):
            bucket = "both_at_reference"
        elif all(
            side.sample_gates[index][side.best_index(index)] != "ok"
            for side in (left, right)
        ):
            bucket = "both_failed"
        else:
            bucket = "both_partial"
        buckets[bucket].append(prompt.id)
    return {
        "ties": ties,
        "buckets": {name: {"n": len(ids), "ids": ids} for name, ids in buckets.items()},
    }


def held_out_prompts(limit: int | None = None) -> list[Prompt]:
    return select(load_prompts(), split=EVAL, limit=limit)


def write_report(report: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return target


def render(report: dict) -> str:
    """The console form: what a reader needs without opening the JSON."""
    lines = [
        f"held-out prompts: {report['prompts']['n']}  "
        + "  ".join(f"{k}={v}" for k, v in report["prompts"]["by_task"].items()),
        "",
        f"{'system':<12} {'best-of-N':>10} {'mean-of-N':>10} {'median':>9} "
        f"{'distinct':>9} {'jaccard':>8}",
    ]
    for name, entry in report["systems"].items():
        diversity = entry.get("diversity") or {}
        lines.append(
            f"{name:<12} {entry['best_of_n'].get('mean', 0):>10.3f} "
            f"{entry['mean_of_n'].get('mean', 0):>10.3f} "
            f"{entry['best_of_n'].get('median', 0):>9.3f} "
            f"{diversity.get('mean_distinct_term_sets', float('nan')):>9.2f} "
            f"{diversity.get('mean_pairwise_jaccard', float('nan')):>8.3f}"
        )
    if report["win_rates"]:
        lines += ["", f"{'pairing':<30} {'win':>5} {'loss':>5} {'tie':>5} {'rate':>7}"]
        for pairing, value in report["win_rates"].items():
            lines.append(
                f"{pairing:<30} {value['wins']:>5} {value['losses']:>5} "
                f"{value['ties']:>5} {value['win_rate']:>7.2%}"
            )
    lines += _render_detail(report)
    return "\n".join(lines)


def _render_detail(report: dict) -> list[str]:
    """The typical-answer rates, the per-sample rates, and why ties tied.
    Printed only when a sampled system is present; deterministic systems have
    one answer, so their typical and best rates are the same numbers."""
    systems = report["systems"]
    sampled = [name for name, entry in systems.items() if entry.get("sampled")]
    if not sampled:
        return []
    lines = ["", f"{'typical answer (mean-of-N)':<30} {'win':>5} {'loss':>5} {'tie':>5} {'rate':>7}"]
    typical = report.get("win_rates_typical", {})
    for challenger in sampled:
        for incumbent in systems:
            value = typical.get(f"{challenger}_vs_{incumbent}")
            if value is None:
                continue
            lines.append(
                f"{challenger + '_vs_' + incumbent:<30} {value['wins']:>5} "
                f"{value['losses']:>5} {value['ties']:>5} {value['win_rate']:>7.2%}"
            )
    lines += ["", f"{'per sample':<12} {'usable':>8} {'at-ref':>8}"]
    for name in sampled:
        entry = systems[name]
        at_ref = entry.get("at_reference_rate")
        lines.append(
            f"{name:<12} {entry.get('usable_rate', float('nan')):>8.2%} "
            + (f"{at_ref:>8.2%}" if at_ref is not None else f"{'-':>8}")
        )
    for pairing, value in report.get("tie_breakdown", {}).items():
        counts = ", ".join(f"{k} {v['n']}" for k, v in value["buckets"].items())
        lines += ["", f"ties {pairing}: {value['ties']} = {counts}"]
    return lines
