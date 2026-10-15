#!/usr/bin/env python3
"""Merge a release adapter into its base model and write one GGUF file.

    python tools/export_gguf.py \\
        --adapter release/re-v3-3b-ck100-20260923-0303/adapter \\
        --out release/re-v3-3b-ck100-20260923-0303/gguf --quant Q8_0

Three steps, each in its own process so the one before it has handed its
memory back to the machine before the next one starts (a 16 GB Mac holds one
fp16 3B model comfortably and two not at all):

1. Load the base model in fp16 on the CPU, put the adapter on it, merge the
   LoRA weights in with ``merge_and_unload``, and save the merged model and the
   base tokenizer to ``OUT/merged-hf``. Never MPS: the merge is a one-off and
   the unified memory is the constraint, not the speed.
2. Convert ``merged-hf`` with llama.cpp's ``convert_hf_to_gguf.py`` straight to
   the requested type (the script writes Q8_0 and F16 itself), into
   ``OUT/model-<QUANT>.gguf``. Homebrew's llama.cpp does not ship the script, so
   ``--llama-cpp`` names a checkout and a shallow clone is made there if the
   script is absent. The script needs the ``gguf`` package in this venv.
3. Check the GGUF carries the tokenizer's own chat template, write
   ``OUT/EXPORT.json`` (adapter label, base model id, quant, each file's size
   and SHA-256, the llama.cpp commit, the step wall times, the timestamp), and
   delete ``merged-hf`` unless ``--keep-merged``.

The adapter label follows ``tools/serve_sidecar.py``: the release directory's
name for ``.../<release>/adapter``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
CONVERTER = "convert_hf_to_gguf.py"
QUANTS = {"Q8_0": "q8_0", "F16": "f16"}
MERGED = "merged-hf"


def adapter_label(path: str | Path) -> str:
    """The release directory's name for ``.../<release>/adapter``, else the basename."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name == "adapter" and resolved.parent.name:
        return resolved.parent.name
    return resolved.name


def gguf_name(quant: str) -> str:
    return f"model-{quant}.gguf"


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Step 1: merge, in a child process
# ---------------------------------------------------------------------------


def merge(model_id: str, adapter: Path, merged_dir: Path) -> None:
    """Base in fp16 on CPU, adapter merged in, saved with the base tokenizer."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(model_id).save_pretrained(str(merged_dir))


# ---------------------------------------------------------------------------
# Step 2: convert, with llama.cpp's own script
# ---------------------------------------------------------------------------


def ensure_llama_cpp(checkout: Path) -> Path:
    """The converter script in ``checkout``, cloning llama.cpp there if absent."""
    script = checkout / CONVERTER
    if not script.is_file():
        if checkout.exists() and any(checkout.iterdir()):
            raise SystemExit(f"{checkout} exists but has no {CONVERTER}; point --llama-cpp elsewhere")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(checkout)], check=True
        )
    if not script.is_file():
        raise SystemExit(f"no {CONVERTER} in {checkout} after cloning")
    return script


def llama_cpp_commit(checkout: Path) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return done.stdout.strip() if done.returncode == 0 else None


def convert_command(script: Path, merged_dir: Path, out_file: Path, quant: str, name: str) -> list[str]:
    return [
        sys.executable,
        str(script),
        str(merged_dir),
        "--outtype",
        QUANTS[quant],
        "--outfile",
        str(out_file),
        "--model-name",
        name,
    ]


# ---------------------------------------------------------------------------
# Step 3: check and record
# ---------------------------------------------------------------------------


def gguf_chat_template(path: Path) -> str | None:
    """``tokenizer.chat_template`` from a GGUF's metadata, or None."""
    from gguf import GGUFReader

    field = GGUFReader(str(path)).get_field("tokenizer.chat_template")
    if field is None:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def hf_chat_template(merged_dir: Path) -> str | None:
    """The chat template the saved tokenizer carries, from either file it may use."""
    jinja = merged_dir / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config = merged_dir / "tokenizer_config.json"
    if config.is_file():
        return json.loads(config.read_text(encoding="utf-8")).get("chat_template")
    return None


def export_record(
    *,
    adapter: Path,
    model_id: str,
    quant: str,
    files: list[Path],
    llama_cpp: str | None,
    seconds: dict[str, float],
    chat_template_matches: bool,
) -> dict:
    return {
        "adapter": adapter_label(adapter),
        "adapter_path": str(adapter),
        "base_model": model_id,
        "quant": quant,
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        ],
        "chat_template_matches_tokenizer": chat_template_matches,
        "llama_cpp_commit": llama_cpp,
        "seconds": {key: round(value, 1) for key, value in seconds.items()},
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--adapter", required=True, help="the LoRA adapter directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the base model id")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--quant", choices=sorted(QUANTS), default="Q8_0")
    parser.add_argument(
        "--llama-cpp",
        default=os.environ.get("LLAMA_CPP_DIR"),
        help="a llama.cpp checkout holding convert_hf_to_gguf.py, cloned there if absent "
        "(default: $LLAMA_CPP_DIR)",
    )
    parser.add_argument("--keep-merged", action="store_true", help=f"keep OUT/{MERGED}")
    parser.add_argument("--merge-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.merge_only and not args.llama_cpp:
        parser.error("--llama-cpp (or $LLAMA_CPP_DIR) is required")
    return args


def _say(message: str) -> None:
    print(f"export_gguf: {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = Path(args.adapter).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    merged_dir = out / MERGED

    if args.merge_only:
        merge(args.model, adapter, merged_dir)
        return 0

    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"no adapter_config.json in {adapter}")
    target = out / gguf_name(args.quant)
    if target.exists():
        raise SystemExit(f"{target} already exists; remove it first")
    out.mkdir(parents=True, exist_ok=True)
    script = ensure_llama_cpp(Path(args.llama_cpp).expanduser().resolve())
    seconds: dict[str, float] = {}

    started = time.monotonic()
    if (merged_dir / "config.json").is_file():
        _say(f"reusing {merged_dir}")
    else:
        _say(f"merging {adapter_label(adapter)} into {args.model} on CPU, fp16")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--merge-only",
             "--adapter", str(adapter), "--model", args.model, "--out", str(out)],
            check=True,
        )
    seconds["merge"] = time.monotonic() - started
    _say(f"merged in {seconds['merge']:.0f} s")

    started = time.monotonic()
    _say(f"converting to {args.quant}")
    subprocess.run(
        convert_command(script, merged_dir, target, args.quant, adapter_label(adapter)),
        check=True,
    )
    seconds["convert"] = time.monotonic() - started
    _say(f"converted in {seconds['convert']:.0f} s: {target}")

    expected = hf_chat_template(merged_dir)
    written = gguf_chat_template(target)
    matches = expected is not None and written is not None and written.strip() == expected.strip()
    if not matches:
        raise SystemExit(
            f"{target.name} does not carry the tokenizer's chat template; "
            f"llama-server would render a different prompt than training did"
        )

    started = time.monotonic()
    record = export_record(
        adapter=adapter,
        model_id=args.model,
        quant=args.quant,
        files=[target],
        llama_cpp=llama_cpp_commit(script.parent),
        seconds=seconds,
        chat_template_matches=matches,
    )
    record["seconds"]["hash"] = round(time.monotonic() - started, 1)
    (out / "EXPORT.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not args.keep_merged:
        shutil.rmtree(merged_dir)
    _say(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
