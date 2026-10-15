"""The Colab cells, as one runnable file, so the notebook is not the source.

A notebook is a bad place to keep logic and a good place to keep an order of
operations. This module holds the cells; ``notebooks/colab_grpo_first_run.ipynb``
calls them in order and shows their output. Running

    python -m cloris_boolean_rl.train.colab_first_run --all

reproduces the whole first run outside a notebook, which is how it gets
debugged.

The cells, in the order the operator runs them:

  1. ``cell_install``   — pin the stack on the T4 runtime.
  2. ``cell_checkout``  — clone or pull this repo, and prove the vendored
                          scorer still matches its recorded hashes.
  3. ``cell_verify``    — run the test suite. A reward that fails its own
                          tests is not a reward, and finding that out after
                          six GPU hours is the avoidable version of this.
  4. ``cell_baselines`` — the deterministic eval on the held-out prompts.
                          Costs no GPU and prints the headroom the run is
                          trying to take.
  5. ``cell_smoke``     — two GRPO steps end to end on the small model.
  6. ``cell_train``     — the run: 1.5B, 4-bit, LoRA r=16, G=8, ~300 prompts.
  7. ``cell_eval``      — base and checkpoint over the held-out prompts,
                          eight samples each, into one JSON.
  8. ``cell_release``   — zip the adapter and the eval into a release file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPO_URL = "https://github.com/sam-vangelos/boolean-search-rl.git"
DRIVE_DIR = "/content/drive/MyDrive/boolean-search-rl"

CELL_INSTALL = """\
# Cell 1 — the stack. Unsloth pulls a matched torch/transformers set for the
# T4. On its own it resolves TRL to 0.24.0, whose GRPOTrainer import dies on
# mergekit and llm_blender, so TRL is pinned on top to 1.13.0, the version
# the CPU smoke runs on.
!pip -q install "unsloth[colab-new]" "trl>=0.14" "peft>=0.13" "datasets>=3.0"
!pip -q install "trl==1.13.0" "safetensors>=0.8" "pydantic>=2.12,<3"
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
"""

CELL_CHECKOUT = f"""\
# Cell 2 — the rig, and Drive for checkpoints.
#
# For a private fork, optionally add a GitHub fine-grained token in Colab's
# Secrets panel (the key icon on the left) as GITHUB_TOKEN with notebook access
# on. Otherwise the public repository is cloned directly, or unpacked from a
# bundle at {DRIVE_DIR}/rig.zip (`python -m
# cloris_boolean_rl.train.colab_first_run --bundle`).
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p {DRIVE_DIR}
try:
    from google.colab import userdata; TOKEN = userdata.get('GITHUB_TOKEN')
except Exception:
    TOKEN = None
CLONE_URL = "{REPO_URL}".replace("https://", f"https://x-access-token:{{TOKEN}}@") if TOKEN else "{REPO_URL}"
print("clone via", "GITHUB_TOKEN secret" if TOKEN else "public HTTPS (or rig.zip fallback)")
!git clone -q $CLONE_URL /content/boolean-search-rl 2>/dev/null \\
  || (cd /content/boolean-search-rl 2>/dev/null && git pull -q $CLONE_URL main 2>/dev/null) \\
  || (mkdir -p /content/boolean-search-rl && unzip -oq {DRIVE_DIR}/rig.zip -d /content/boolean-search-rl)
%cd /content/boolean-search-rl
!git remote set-url origin {REPO_URL} 2>/dev/null || true
!ls cloris_boolean_rl/scorer/VENDORED_FROM.json && python -c "import json;d=json.load(open('cloris_boolean_rl/scorer/VENDORED_FROM.json'));print('scorer vendored at', d['commit'])"
"""

CELL_VERIFY = """\
# Cell 3 — the reward has to pass its own tests before it pays for anything.
!pip -q install pytest
!python -m pytest tests/ -q
!python tools/build_split.py --check && python tools/build_prompts.py --check
"""

CELL_BASELINES = """\
# Cell 4 — what the held-out prompts are worth before any training.
!python tools/run_eval.py --out runs/eval/before.json
"""

CELL_SMOKE = '''\
# Cell 5 — two steps end to end, then the question that decides the run.
!python -m cloris_boolean_rl.train.grpo --smoke

# GRPO learns from the SPREAD inside the group of G completions. If the base
# model cannot produce a legal single OR group at all, every sample floors,
# reward_std is zero, and 100 steps buy nothing — which is exactly what the
# 0.5B smoke shows on CPU (reward -13.82, reward_std 0, frac_reward_zero_std 1).
# So measure the 1.5B's contract pass rate on eight held-out prompts before
# spending the hours.
!python tools/run_eval.py --model Qwen/Qwen2.5-1.5B-Instruct \\
    --limit 8 --samples 8 --temperature 0.9 --out runs/eval/base_probe.json
import json
probe = json.load(open("runs/eval/base_probe.json"))["systems"]["base"]
print(json.dumps(probe["gates"], indent=2), probe["diversity"])
ok = probe["gates"].get("ok", 0)
print(f"\\ncontract pass rate: {ok}/8 prompts have a legal best sample")
print("STOP and add a format warm-up if this is 0" if ok == 0 else "there is a gradient; continue")
'''

CELL_TRAIN = f"""\
# Cell 6 — the run. ~300 prompts, G=8, one epoch, checkpoints to Drive.
# Budget: a T4 generates roughly 16x160 tokens a step; 150 steps is several
# hours. Start with --max-steps 100 and resume from a checkpoint if Colab
# drops the session. Unsloth runs with fp16 off; grpo.py says why.
# A dropped session resumes with --resume-from runs/re-v1/checkpoint-<N>.
!python -m cloris_boolean_rl.train.grpo \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --prompts 300 --generations 8 --epochs 1 --max-steps 100 \\
    --temperature 0.9 --beta 0.04 --save-steps 25 \\
    --output-dir runs/re-v1
!cp -r runs/re-v1 {DRIVE_DIR}/
# Run 2 (2026-09-22) used --learning-rate 5e-5 --warmup-steps 10 --lr-scheduler cosine --grad-accum 4 --max-steps 200 --max-completion-length 128; see the plan addendum.
"""

CELL_EVAL = """\
# Cell 7 — base against checkpoint, eight samples each, held-out prompts only.
!python tools/run_eval.py \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --adapter runs/re-v1/adapter \\
    --samples 8 --temperature 0.8 \\
    --out runs/eval/after.json \\
    --note "first GRPO run, re-v1 split"
import json; print(json.dumps(json.load(open('runs/eval/after.json'))['win_rates'], indent=2))
"""

CELL_RELEASE = f"""\
# Cell 8 — the hand-back. One zip: adapter, eval JSON, run config, split.
!mkdir -p release/re-v1 && cp -r runs/re-v1/adapter release/re-v1/ \\
  && cp runs/re-v1/run_config.json runs/re-v1/train_summary.json release/re-v1/ \\
  && cp runs/eval/before.json runs/eval/after.json release/re-v1/ \\
  && cp cloris_boolean_rl/data/splits/re-v1.json release/re-v1/split.json \\
  && cp cloris_boolean_rl/scorer/VENDORED_FROM.json release/re-v1/
!cd release && zip -qr cloris-boolean-rl-re-v1.zip re-v1 && ls -la
!cp release/cloris-boolean-rl-re-v1.zip {DRIVE_DIR}/
"""

CELLS = [
    ("install", CELL_INSTALL),
    ("checkout", CELL_CHECKOUT),
    ("verify", CELL_VERIFY),
    ("baselines", CELL_BASELINES),
    ("smoke", CELL_SMOKE),
    ("train", CELL_TRAIN),
    ("eval", CELL_EVAL),
    ("release", CELL_RELEASE),
]


# ---------------------------------------------------------------------------
# The same steps, runnable outside a notebook
# ---------------------------------------------------------------------------


def _run(*command: str) -> int:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.call(list(command), cwd=ROOT)


def cell_verify() -> int:
    return (
        _run(sys.executable, "-m", "pytest", "tests/", "-q")
        or _run(sys.executable, "tools/build_split.py", "--check")
        or _run(sys.executable, "tools/build_prompts.py", "--check")
    )


def cell_baselines() -> int:
    return _run(sys.executable, "tools/run_eval.py", "--out", "runs/eval/before.json")


def cell_smoke() -> int:
    return _run(sys.executable, "-m", "cloris_boolean_rl.train.grpo", "--smoke")


def cell_train(max_steps: int = 100) -> int:
    return _run(
        sys.executable, "-m", "cloris_boolean_rl.train.grpo",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--prompts", "300", "--generations", "8", "--max-steps", str(max_steps),
        "--temperature", "0.9", "--beta", "0.04", "--save-steps", "25",
        "--output-dir", "runs/re-v1",
    )


def cell_eval() -> int:
    return _run(
        sys.executable, "tools/run_eval.py",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--adapter", "runs/re-v1/adapter",
        "--samples", "8", "--temperature", "0.8",
        "--out", "runs/eval/after.json",
    )


def cell_release() -> int:
    stage = ROOT / "release" / "re-v1"
    stage.mkdir(parents=True, exist_ok=True)
    wanted = [
        ROOT / "runs/re-v1/adapter",
        ROOT / "runs/re-v1/run_config.json",
        ROOT / "runs/re-v1/train_summary.json",
        ROOT / "runs/eval/before.json",
        ROOT / "runs/eval/after.json",
        ROOT / "cloris_boolean_rl/data/splits/re-v1.json",
        ROOT / "cloris_boolean_rl/scorer/VENDORED_FROM.json",
    ]
    for path in wanted:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        target = stage / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
    archive = shutil.make_archive(
        str(ROOT / "release" / "cloris-boolean-rl-re-v1"), "zip", stage.parent, "re-v1"
    )
    print(f"release: {archive}")
    return 0


BUNDLE_EXCLUDE = (".git", ".venv", "runs", "release", "__pycache__", ".pytest_cache")


def make_bundle(path: Path | None = None) -> Path:
    """Zip the rig for a Colab session that cannot clone it.

    This repo has no git remote, so cell 2's clone fails and falls back to
    unpacking this file from Drive. Everything the run needs is in it —
    package, tests, tools, the split, the prompt set, the vendored scorer —
    and everything it does not is excluded, which is what keeps it small
    enough to upload.
    """
    import zipfile

    target = Path(path or ROOT / "release" / "rig.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(ROOT.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(ROOT)
            if any(part in BUNDLE_EXCLUDE for part in relative.parts):
                continue
            if relative.suffix in {".pyc", ".zip"}:
                continue
            bundle.write(item, relative)
            written += 1
    size = target.stat().st_size / 1e6
    print(f"bundle: {target}  ({written} files, {size:.1f} MB)")
    print(f"upload it to Drive at {DRIVE_DIR}/rig.zip, then run cell 2")
    return target


def write_notebook(path: Path) -> Path:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Stage 3, first GRPO run\n",
                    "\n",
                    "Runtime: **T4 GPU** (Runtime -> Change runtime type -> T4). ",
                    "Run the cells in order; cell 3 has to be green before cell 6 ",
                    "is worth the hours.\n",
                ],
            }
        ]
        + [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"name": name},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
            for name, source in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-notebook", action="store_true")
    parser.add_argument("--bundle", action="store_true", help="zip the rig for Colab")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--print", dest="show", action="store_true")
    args = parser.parse_args(argv)

    if args.bundle:
        make_bundle()
        return 0
    if args.write_notebook:
        print(write_notebook(ROOT / "notebooks" / "colab_grpo_first_run.ipynb"))
        return 0
    if args.show or not any(vars(args).values()):
        for name, source in CELLS:
            print(f"# ---- {name} " + "-" * (60 - len(name)))
            print(source)
        return 0

    status = 0
    if args.verify or args.all:
        status = status or cell_verify()
    if args.baselines or args.all:
        status = status or cell_baselines()
    if args.smoke or args.all:
        status = status or cell_smoke()
    if args.train or args.all:
        status = status or cell_train(args.max_steps)
    if args.eval or args.all:
        status = status or cell_eval()
    if args.release or args.all:
        status = status or cell_release()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
