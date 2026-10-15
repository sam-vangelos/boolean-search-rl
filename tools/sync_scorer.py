#!/usr/bin/env python3
"""Re-copy Cloris's scorer into this repo, and say exactly what changed.

    python tools/sync_scorer.py --check          # report drift, write nothing
    python tools/sync_scorer.py --apply          # re-copy and update the note
    python tools/sync_scorer.py --cloris <path>  # a worktree other than the default

The scorer is the reward. If the vendored copy drifts from the one Cloris
ships, a policy is being trained against a judge that no longer exists, and
every number the eval prints is about a scorer nobody runs. So drift is a
failure by default: ``--check`` exits non-zero on any difference, prints the
unified diff, and refuses to guess whether the change was wanted.

The reverse direction is not supported on purpose. Nothing in this repo may
edit the scorer; a fix belongs in Cloris and comes back through this tool.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "cloris_boolean_rl" / "scorer"
NOTE = VENDOR / "VENDORED_FROM.json"
_ENV_CLORIS = os.environ.get("SOURCING_AGENT_ROOT") or os.environ.get("CLORIS_ROOT")
DEFAULT_CLORIS = Path(_ENV_CLORIS) if _ENV_CLORIS else None
SOURCE_SUBPATH = "linkedin/boolean_score"

FILES = (
    "__init__.py",
    "parser.py",
    "lexicon.py",
    "score.py",
    "enrich.py",
    "lexicons/research-engineer-post-training.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compare(source_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """(identical, changed, missing)."""
    identical, changed, missing = [], [], []
    for name in FILES:
        src, dst = source_dir / name, VENDOR / name
        if not src.exists() or not dst.exists():
            missing.append(name)
        elif digest(src) == digest(dst):
            identical.append(name)
        else:
            changed.append(name)
    return identical, changed, missing


def show_diff(source_dir: Path, name: str) -> None:
    src, dst = source_dir / name, VENDOR / name
    if name.endswith(".json"):
        print(f"--- {name}: {dst.stat().st_size} bytes here, "
              f"{src.stat().st_size} in Cloris (json, diff suppressed)")
        return
    lines = difflib.unified_diff(
        dst.read_text().splitlines(keepends=True),
        src.read_text().splitlines(keepends=True),
        fromfile=f"vendored/{name}",
        tofile=f"cloris/{name}",
    )
    sys.stdout.writelines(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloris", type=Path, default=DEFAULT_CLORIS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.cloris is None:
        print(
            "pass --cloris <path> or set SOURCING_AGENT_ROOT / CLORIS_ROOT",
            file=sys.stderr,
        )
        return 2

    source_dir = args.cloris / SOURCE_SUBPATH
    if not source_dir.is_dir():
        print(f"no scorer at {source_dir}", file=sys.stderr)
        return 2

    recorded = json.loads(NOTE.read_text()) if NOTE.exists() else {}
    current = head_sha(args.cloris)
    identical, changed, missing = compare(source_dir)

    print(f"cloris:   {args.cloris}  HEAD {current}")
    print(f"vendored: recorded at {recorded.get('commit', '(none)')}")
    print(f"  identical {len(identical)}  changed {len(changed)}  missing {len(missing)}")
    for name in changed:
        print(f"  CHANGED  {name}")
    for name in missing:
        print(f"  MISSING  {name}")

    if not args.apply:
        for name in changed:
            show_diff(source_dir, name)
        if changed or missing:
            print("\nvendored scorer has drifted; re-run with --apply", file=sys.stderr)
            return 1
        print("vendored scorer matches Cloris byte for byte")
        return 0

    for name in FILES:
        src, dst = source_dir / name, VENDOR / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    NOTE.write_text(
        json.dumps(
            {
                **recorded,
                "source_repo": "long-horizon-sourcing-agent",
                "source_path": SOURCE_SUBPATH,
                "commit": current,
                "vendored_on": date.today().isoformat(),
                "files": list(FILES),
                "sha256": {name: digest(VENDOR / name) for name in FILES},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nre-copied {len(FILES)} files at {current}; VENDORED_FROM.json updated")
    if changed or missing:
        print("RUN THE TESTS: the reward just changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
