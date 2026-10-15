#!/usr/bin/env python3
"""Serve the trained adapter to Cloris over HTTP on 127.0.0.1.

    python tools/serve_sidecar.py --adapter release/re-v3-3b-ck100-20260923-0303/adapter
    python tools/serve_sidecar.py --gguf release/re-v3-3b-ck100-20260923-0303/gguf/model-Q8_0.gguf
    python tools/serve_sidecar.py --fake --port 8799        # canned answers, no torch

``POST /enrich`` returns candidate OR groups for one register or anchor
request; ``GET /health`` says which model, adapter and device are answering.
The adapter is reported by its release directory's name, so every response
Cloris records carries the version that produced it. ``--fake`` serves
deterministic canned candidates without importing torch, for Cloris's smoke
test and for machines with no model.

``--gguf`` is the Mac's backend: the merged model ``tools/export_gguf.py``
wrote, served by llama.cpp's ``llama-server`` on Metal, which this starts on
127.0.0.1:``--llama-port`` and stops on shutdown. Its label is the release
directory's name for ``.../<release>/gguf/model-Q8_0.gguf``.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloris_boolean_rl.serve import sidecar  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FAKE = "fake"


def adapter_label(path: str) -> str:
    """The release directory's name for ``.../<release>/adapter``, else the basename."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name == "adapter" and resolved.parent.name:
        return resolved.parent.name
    return resolved.name


def gguf_label(path: str) -> str:
    """The release directory's name for ``.../<release>/gguf/<file>.gguf``, else the file's stem."""
    resolved = Path(path).expanduser().resolve()
    if resolved.parent.name == "gguf" and resolved.parent.parent.name:
        return resolved.parent.parent.name
    return resolved.stem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--adapter", default=None, help="the LoRA adapter directory, on transformers")
    backend.add_argument("--gguf", default=None, help="a merged GGUF, on llama-server (Metal)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the base model id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--llama-port",
        type=int,
        default=sidecar.DEFAULT_LLAMA_PORT,
        help="where the spawned llama-server listens, with --gguf",
    )
    parser.add_argument("--fake", action="store_true", help="canned candidates, no model, no torch")
    parser.add_argument("--verbose", action="store_true", help="log every request to stderr")
    args = parser.parse_args(argv)
    if not args.fake and not (args.adapter or args.gguf):
        parser.error("--adapter or --gguf is required unless --fake is given")
    return args


def build_generator(args: argparse.Namespace) -> tuple[object, dict[str, str]]:
    """The generator and what it reports, loading the model unless ``--fake``."""
    if args.fake:
        return sidecar.fake_generator, {"model": FAKE, "adapter": FAKE, "device": FAKE}
    if args.gguf:
        generator = sidecar.load_llama_generator(args.gguf, port=args.llama_port)
        label = gguf_label(args.gguf)
    else:
        generator = sidecar.load_generator(args.model, args.adapter, args.device)
        label = adapter_label(args.adapter)
    return generator, {"model": args.model, "adapter": label, "device": generator.device}


def make_handler(generator, info: dict[str, str], args: argparse.Namespace) -> type:
    return sidecar.make_handler(
        generator,
        model_id=info["model"],
        adapter_label=info["adapter"],
        device=info["device"],
        verbose=args.verbose,
    )


def build(args: argparse.Namespace) -> tuple[type, dict[str, str]]:
    """The handler class and what it reports, loading the model unless ``--fake``."""
    generator, info = build_generator(args)
    return make_handler(generator, info, args), info


def close(generator) -> None:
    """Stop whatever the generator started (llama-server); a no-op for the others."""
    closer = getattr(generator, "close", None)
    if callable(closer):
        closer()


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Ctrl-C and a plain `kill` both stop it cleanly. A shell that starts a
    # background job leaves SIGINT ignored, so both are installed explicitly.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        generator, info = build_generator(args)
    except KeyboardInterrupt:
        return 130
    handler = make_handler(generator, info, args)

    def announce(url: str) -> None:
        print(
            f"sidecar: model={info['model']} adapter={info['adapter']} "
            f"device={info['device']} url={url}",
            flush=True,
        )

    try:
        sidecar.serve(args.host, args.port, handler, on_ready=announce)
    finally:
        close(generator)
    print("sidecar: stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
