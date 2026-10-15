#!/usr/bin/env python3
"""Score every system on the held-out prompts and write eval JSON.

    python tools/run_eval.py                              # no model: bare, enricher, reference
    python tools/run_eval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 8
    python tools/run_eval.py --model <base> --adapter runs/re-v1/checkpoint-100

The deterministic half needs neither torch nor a network and is what says
how much headroom Stage 3 actually has. Pass ``--model`` to add the sampled
baseline, and ``--adapter`` to add a checkpoint beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloris_boolean_rl.eval import baselines, harness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", choices=["register", "anchor"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--samples", type=int, default=harness.DEFAULT_SAMPLES)
    parser.add_argument("--temperature", type=float, default=harness.DEFAULT_TEMPERATURE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="runs/eval/latest.json")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)

    from cloris_boolean_rl.data.prompts import EVAL, load_prompts, select

    prompts = select(load_prompts(), split=EVAL, task=args.task, limit=args.limit)
    if not prompts:
        print("no held-out prompts matched", file=sys.stderr)
        return 1

    systems = harness.run_deterministic(prompts)

    if args.model:
        from cloris_boolean_rl.eval.generate import load_model, make_generator

        model, tokenizer = load_model(args.model, device=args.device)
        systems[baselines.BASE] = harness.run_sampled(
            baselines.BASE,
            prompts,
            make_generator(model, tokenizer),
            samples=args.samples,
            temperature=args.temperature,
        )
        if args.adapter:
            from peft import PeftModel

            tuned = PeftModel.from_pretrained(model, args.adapter).eval()
            systems["checkpoint"] = harness.run_sampled(
                "checkpoint",
                prompts,
                make_generator(tuned, tokenizer),
                samples=args.samples,
                temperature=args.temperature,
            )

    report = harness.build_report(
        prompts,
        systems,
        note=args.note or (f"model={args.model} adapter={args.adapter}" if args.model else "deterministic only"),
    )
    path = harness.write_report(report, ROOT / args.out)
    print(harness.render(report))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
