#!/usr/bin/env python3
"""Write the prompt set to data/prompts/re-v1.json and print the counts.

    python tools/build_prompts.py [--check] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloris_boolean_rl.data import prompts as prompts_mod  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args(argv)

    built = prompts_mod.build_prompts()

    if args.check:
        if not prompts_mod.PROMPT_PATH.exists():
            print("no prompt set written", file=sys.stderr)
            return 1
        saved = json.loads(prompts_mod.PROMPT_PATH.read_text())
        if [p["id"] for p in saved["prompts"]] != [p.id for p in built]:
            print("saved prompt set has drifted", file=sys.stderr)
            return 1
        print(f"prompt set is current: {len(built)} prompts")
        return 0

    path = prompts_mod.write_prompts(built)
    print(f"wrote {path.relative_to(ROOT)}  ({len(built)} prompts)")
    for key, value in prompts_mod.counts(built).items():
        print(f"  {key:44s} {value:5d}")

    for index in range(args.sample):
        prompt = built[index * max(1, len(built) // max(1, args.sample))]
        print(f"\n--- {prompt.id} [{prompt.split}/{prompt.task}/{prompt.source}]")
        print(f"given:     {prompt.given}")
        print(f"reference: {prompt.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
