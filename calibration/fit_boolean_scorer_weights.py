#!/usr/bin/env python3
"""Fit the scorer's weights to the blind-rater targets, with a held-out check.

Eight slices of reading residuals and writing rules have plateaued at a rank
correlation of 0.61 to 0.64 against raters who agree with each other at 0.886.
Ablating the bloat terms one at a time showed why that is hard to push by
hand: two of them are anti-correlated with the raters, three contribute
nothing, and the whole structure component moves the correlation by at most
0.0012 while holding a tenth of the weighted mean. The next decision needs a
fit.

This tool changes nothing. It exports one feature row per string, fits three
models, cross-validates each one, checks every fit against the acceptance
guardrails, and prints the coefficient tables. Nothing here is wired into
linkedin/boolean_score.

    python calibration/fit_boolean_scorer_weights.py [--out <feature export path>]

Standard library only. The fits are coordinate descent on squared error, which
is enough for a dozen parameters and avoids a dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
_ENV_CLORIS = os.environ.get("SOURCING_AGENT_ROOT") or os.environ.get("CLORIS_ROOT")
if _ENV_CLORIS:
    if _ENV_CLORIS not in sys.path:
        sys.path.insert(0, _ENV_CLORIS)
    from linkedin.boolean_score.lexicon import LEXICON_DIR, load_lexicon  # noqa: E402
    from linkedin.boolean_score.score import (  # noqa: E402
        BLOAT_COEFFICIENTS,
        NEUTRAL_SCORE,
        STRUCTURE_WEIGHTS,
        UNMAPPED_RELEVANCE,
        WEIGHTS,
        score,
    )
    FIXTURES = Path(_ENV_CLORIS) / "tests" / "fixtures" / "boolean_corpus"
else:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from cloris_boolean_rl.scorer.lexicon import LEXICON_DIR, load_lexicon  # noqa: E402
    from cloris_boolean_rl.scorer.score import (  # noqa: E402
        BLOAT_COEFFICIENTS,
        NEUTRAL_SCORE,
        STRUCTURE_WEIGHTS,
        UNMAPPED_RELEVANCE,
        WEIGHTS,
        score,
    )
    FIXTURES = ROOT / "cloris_boolean_rl" / "data" / "fixtures"

LEXICON_PATH = LEXICON_DIR / "research-engineer-post-training.json"
DEFAULT_EXPORT = Path(__file__).resolve().parent / "feature_export.json"

COVERAGE_KEYS = tuple(WEIGHTS)
BLOAT_KEYS = tuple(sorted(BLOAT_COEFFICIENTS))
STRUCTURE_KEYS = tuple(sorted(STRUCTURE_WEIGHTS))
SHAPE_KEYS = ("anchor_present", "group_count", "term_count", "char_length", "mapped_fraction")

FOLDS = 5
REPEATS = 10
EXEMPLAR = (
    '("evaluation" OR "evaluations" OR "eval" OR "evals") AND '
    '("agentic coding" OR "software engineering agents" OR "coding agents" OR "code gen")'
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _features(boolean: str, lexicon) -> dict[str, Any]:
    report = score(boolean, lexicon)
    groups = [group for group in report.groups if not group.negated]
    terms = [term for group in report.groups for term in group.terms]
    mapped = sum(1 for term in terms if lexicon.concept_for(term.text) is not None)
    row: dict[str, Any] = {
        "composite": report.composite,
        "legal": report.legal,
        "anchor_present": 1.0 if report.anchor_present else 0.0,
        "group_count": float(len(groups)),
        "term_count": float(len(terms)),
        "char_length": float(len(boolean)),
        "mapped_fraction": (mapped / len(terms)) if terms else 0.0,
    }
    readable = any(group.mapped_ids for group in report.groups)
    neutral = NEUTRAL_SCORE if readable else UNMAPPED_RELEVANCE
    for name in COVERAGE_KEYS:
        value = report.components.get(name)
        row[f"applicable_{name}"] = 0.0 if value is None else 1.0
        # N/A is filled with the neutral the scorer itself substitutes for
        # coverage; cohesion has no neutral (the scorer drops it), so it is
        # filled with the same 0.5 and the applicability flag carries the truth.
        row[name] = neutral if value is None else float(value)
    for key in BLOAT_KEYS:
        row[f"bloat_{key}"] = float(report.bloat.get(key, 0.0))
    for key in STRUCTURE_KEYS:
        row[f"structure_{key}"] = float(report.structure_charges.get(key, 0.0))
    return row


def build_export(lexicon) -> dict[str, list[dict[str, Any]]]:
    targets = json.loads((FIXTURES / "rater_targets.json").read_text())["strings"]
    corpus = json.loads((FIXTURES / "ml_runs_76.json").read_text())["strings"]
    adversarial = json.loads((FIXTURES / "adversarial.json").read_text())["strings"]
    in_role = {row["id"] for row in targets}

    blocks: dict[str, list[dict[str, Any]]] = {"targets": [], "pae": [], "adversarial": [], "corpus": []}
    for row in targets:
        blocks["targets"].append(
            {"id": row["id"], "target": row["target"], "fable": row["fable"],
             "opus": row["opus"], **_features(row["boolean"], lexicon)}
        )
    for row in corpus:
        features = {"id": row["id"], **_features(row["boolean"], lexicon)}
        blocks["corpus"].append(features)
        if row["id"] not in in_role:
            blocks["pae"].append(features)
    for row in adversarial:
        blocks["adversarial"].append(
            {"id": row["id"], "defect": row["defect"], **_features(row["boolean"], lexicon)}
        )
    blocks["exemplar"] = [{"id": "exemplar", **_features(EXEMPLAR, lexicon)}]
    return blocks


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    out = [0.0] * len(values)
    for position, index in enumerate(order):
        out[index] = float(position)
    return out


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    return statistics.correlation(_ranks(left), _ranks(right))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

Model = Callable[[dict[str, Any]], float]

LINEAR_POSITIVE = COVERAGE_KEYS + ("anchor_present", "mapped_fraction")
LINEAR_NEGATIVE = tuple(f"bloat_{key}" for key in BLOAT_KEYS) + tuple(
    f"structure_{key}" for key in STRUCTURE_KEYS
)


def baseline_model(_rows: Sequence[dict[str, Any]]) -> Model:
    return lambda row: row["composite"]


def fit_linear(rows: Sequence[dict[str, Any]], passes: int = 60) -> tuple[Model, dict[str, float]]:
    names = LINEAR_POSITIVE + LINEAR_NEGATIVE
    bounds = {name: (0.0, 2.0) for name in LINEAR_POSITIVE}
    bounds.update({name: (-2.0, 0.0) for name in LINEAR_NEGATIVE})
    weights = {name: 0.0 for name in names}
    intercept = statistics.fmean(row["target"] for row in rows)
    for _ in range(passes):
        residual = [
            row["target"] - intercept - sum(weights[n] * row[n] for n in names)
            for row in rows
        ]
        intercept += statistics.fmean(residual)
        for name in names:
            column = [row[name] for row in rows]
            denominator = sum(value * value for value in column)
            if denominator <= 1e-12:
                continue
            partial = [
                row["target"] - intercept
                - sum(weights[n] * row[n] for n in names if n != name)
                for row in rows
            ]
            raw = sum(c * p for c, p in zip(column, partial)) / denominator
            low, high = bounds[name]
            weights[name] = max(low, min(high, raw))
    coefficients = {"intercept": intercept, **weights}

    def model(row: dict[str, Any]) -> float:
        value = intercept + sum(weights[n] * row[n] for n in names)
        return max(0.0, min(1.0, value))

    return model, coefficients


def _multiplicative(row: dict[str, Any], cover: dict[str, float], bloat: dict[str, float]) -> float:
    total = sum(cover.values())
    if total <= 0:
        return 0.0
    mean = sum(cover[n] * row[n] for n in COVERAGE_KEYS) / total
    charge = sum(bloat[k] * row[f"bloat_{k}"] for k in BLOAT_KEYS)
    return max(0.0, 1.0 - min(1.0, charge)) * mean


def fit_multiplicative(
    rows: Sequence[dict[str, Any]], passes: int = 8
) -> tuple[Model, dict[str, float], dict[str, float]]:
    cover = {name: WEIGHTS[name] for name in COVERAGE_KEYS}
    bloat = {key: 1.0 for key in BLOAT_KEYS}
    grid = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)

    def error(c: dict[str, float], b: dict[str, float]) -> float:
        return sum((row["target"] - _multiplicative(row, c, b)) ** 2 for row in rows)

    best = error(cover, bloat)
    for _ in range(passes):
        for name in COVERAGE_KEYS:
            keep, current = cover[name], best
            for candidate in grid:
                cover[name] = candidate
                value = error(cover, bloat)
                if value < current:
                    current, keep = value, candidate
            cover[name], best = keep, current
        for key in BLOAT_KEYS:
            keep, current = bloat[key], best
            for candidate in grid:
                bloat[key] = candidate
                value = error(cover, bloat)
                if value < current:
                    current, keep = value, candidate
            bloat[key], best = keep, current
    frozen_cover, frozen_bloat = dict(cover), dict(bloat)
    return (
        lambda row: _multiplicative(row, frozen_cover, frozen_bloat),
        frozen_cover,
        frozen_bloat,
    )


# ---------------------------------------------------------------------------
# Cross-validation and guardrails
# ---------------------------------------------------------------------------


def cross_validate(
    rows: Sequence[dict[str, Any]],
    fit: Callable[[Sequence[dict[str, Any]]], Model],
    folds: int = FOLDS,
    repeats: int = REPEATS,
) -> tuple[float, float]:
    held_out: list[float] = []
    indices = list(range(len(rows)))
    for seed in range(repeats):
        random.Random(seed).shuffle(indices)
        for fold in range(folds):
            test = {indices[position] for position in range(fold, len(rows), folds)}
            train = [rows[i] for i in range(len(rows)) if i not in test]
            check = [rows[i] for i in sorted(test)]
            model = fit(train)
            held_out.append(
                spearman([model(row) for row in check], [row["target"] for row in check])
            )
    model = fit(rows)
    in_sample = spearman([model(row) for row in rows], [row["target"] for row in rows])
    return statistics.fmean(held_out), in_sample


def guardrails(model: Model, blocks: dict[str, list[dict[str, Any]]]) -> list[str]:
    broken: list[str] = []
    exemplar = model(blocks["exemplar"][0])
    corpus = sorted((model(row) for row in blocks["corpus"]), reverse=True)
    if exemplar < 0.85:
        broken.append(f"exemplar {exemplar:.4f} < 0.85")
    ranked = sorted(corpus + [exemplar], reverse=True)
    rank = ranked.index(exemplar) + 1
    if rank > round(0.10 * len(ranked)):
        broken.append(f"exemplar rank {rank}/{len(ranked)} outside the top decile")
    median = statistics.median(corpus)
    above = [
        row["id"] for row in blocks["adversarial"] if model(row) >= median
    ]
    if above:
        broken.append(f"adversarial at or above the median of {median:.4f}: {', '.join(above)}")
    dump = next(
        (row for row in blocks["adversarial"] if "every form" in row["defect"]), None
    )
    if dump is not None and model(dump) >= 0.10:
        broken.append(f"lexicon dump {model(dump):.4f} >= 0.10")
    return broken


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_EXPORT))
    args = parser.parse_args(argv)

    lexicon = load_lexicon(LEXICON_PATH)
    blocks = build_export(lexicon)
    Path(args.out).write_text(json.dumps(blocks, indent=2, ensure_ascii=False) + "\n")
    rows = blocks["targets"]
    print(f"features exported for {sum(len(v) for v in blocks.values())} strings -> {args.out}")
    print(f"fitting on {len(rows)} in-role targets, {FOLDS}-fold x {REPEATS} splits\n")

    results: list[tuple[str, float, float, Model, list[str]]] = []

    held, sample = cross_validate(rows, baseline_model)
    model = baseline_model(rows)
    results.append(("a. current composition", held, sample, model, guardrails(model, blocks)))

    held, sample = cross_validate(rows, lambda train: fit_linear(train)[0])
    model, linear_coefficients = fit_linear(rows)
    results.append(("b. linear, sign-constrained", held, sample, model, guardrails(model, blocks)))

    held, sample = cross_validate(rows, lambda train: fit_multiplicative(train)[0])
    model, cover, bloat = fit_multiplicative(rows)
    results.append(("c. multiplicative, fitted", held, sample, model, guardrails(model, blocks)))

    print("model                         held-out   in-sample   guardrails")
    for label, held, sample, _model, broken in results:
        verdict = "ok" if not broken else "REJECTED"
        print(f"  {label:28s}  {held:7.4f}   {sample:9.4f}   {verdict}")
        for item in broken:
            print(f"      - {item}")

    print("\n(b) linear coefficients")
    for name, value in sorted(linear_coefficients.items(), key=lambda kv: -abs(kv[1])):
        if abs(value) > 1e-6:
            print(f"  {name:34s} {value:+.4f}")
    zeroed = [n for n, v in linear_coefficients.items() if abs(v) <= 1e-6]
    print(f"  driven to zero: {', '.join(sorted(zeroed)) or 'none'}")

    print("\n(c) multiplicative coefficients")
    print("  coverage weights (current -> fitted)")
    for name in COVERAGE_KEYS:
        print(f"    {name:24s} {WEIGHTS[name]:.2f} -> {cover[name]:.2f}")
    print("  bloat multipliers on the shipped coefficient (1.00 = unchanged)")
    for key in BLOAT_KEYS:
        print(f"    {key:24s} {bloat[key]:.2f}")
    dropped = [k for k in BLOAT_KEYS if bloat[k] == 0.0]
    raised = [k for k in BLOAT_KEYS if bloat[k] > 1.0]
    print(f"  driven to zero: {', '.join(dropped) or 'none'}")
    print(f"  driven up:      {', '.join(raised) or 'none'}")

    admissible = [item for item in results if not item[4]]
    best = max(admissible, key=lambda item: item[1]) if admissible else None
    if best is None:
        print("\nno admissible fit; every model broke a guardrail")
        return 0
    label, held, _sample, model, _broken = best
    print(f"\nbest admissible fit: {label} (held-out {held:.4f})")
    print("ten largest residuals")
    targets = json.loads((FIXTURES / "rater_targets.json").read_text())["strings"]
    rationales = {row["id"]: row for row in targets}
    scored = sorted(
        ((model(row) - row["target"], row) for row in rows), key=lambda item: -abs(item[0])
    )
    for gap, row in scored[:10]:
        source = rationales[row["id"]]
        split = " SPLIT" if abs(source["fable"] - source["opus"]) >= 0.15 else ""
        print(f"  {gap:+.3f} fit {model(row):.3f} target {row['target']:.3f}{split}  {row['id']}")
        print(f"        F: {source['fable_rationale'][:150]}")
        print(f"        O: {source['opus_rationale'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
