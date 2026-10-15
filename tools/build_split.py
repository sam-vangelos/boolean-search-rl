#!/usr/bin/env python3
"""Write the A/B concept split to data/splits/re-v1.json and describe it.

    python tools/build_split.py [--check]

``--check`` recomputes and refuses to write, exiting non-zero when the saved
split has drifted from what the seed produces. That is what CI would run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloris_boolean_rl.data import split as split_mod  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    args = parser.parse_args(argv)

    document = split_mod.load_full_lexicon_document()
    plan = split_mod.build_split(document)

    if args.check:
        if not split_mod.SPLIT_PATH.exists():
            print("no split written; run tools/build_split.py", file=sys.stderr)
            return 1
        saved = json.loads(split_mod.SPLIT_PATH.read_text())
        if saved["a"] != list(plan.a_ids) or saved["b"] != list(plan.b_ids):
            print("saved split has drifted from the seeded one", file=sys.stderr)
            return 1
        print(f"split is current: {len(plan.a_ids)} in A, {len(plan.b_ids)} in B")
        return 0

    path = split_mod.write_split(plan)
    held = set(plan.b_ids)
    per_family = Counter()
    total_family = Counter()
    for concept in document["concepts"]:
        family = split_mod.primary_family(concept)
        total_family[family] += 1
        if concept["id"] in held:
            per_family[family] += 1

    print(f"wrote {path.relative_to(ROOT)}")
    print(f"  A {len(plan.a_ids)} concepts, B {len(plan.b_ids)} concepts "
          f"({len(plan.b_ids) / len(document['concepts']):.0%} held out)")
    print(f"  protected (never held out): {len(split_mod.protected_ids(document))}")
    print(f"  {'family':24s} {'total':>5s} {'held':>5s}")
    for family in sorted(total_family):
        print(f"  {family:24s} {total_family[family]:5d} {per_family[family]:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
