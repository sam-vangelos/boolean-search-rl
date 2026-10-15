"""The enricher sidecar: the trained adapter behind two HTTP endpoints.

Cloris, after its deterministic enricher has run, asks this service for
candidate OR groups, scores them with its own scorer, and keeps one only when
it beats what was there. The service therefore does exactly one thing: build
the messages training built, sample from the model, and hand back each raw
completion beside the group ``reward.tasks.extract_group`` reads out of it.
It judges nothing; the scorer on Cloris's side does.

The contract (``plans/sidecar.md``):

    POST /enrich  {"task": "register"|"anchor", "given": str, "n": int,
                   "temperature": float, "seed": int|null}
               -> {"adapter": str, "candidates": [{"text": str, "group": str|null}]}
    GET  /health  -> {"ok": true, "model": str, "adapter": str, "device": str}

Errors are JSON with a ``detail`` field. Standard library only: one model,
one lock around generation, ``ThreadingHTTPServer`` so a slow request does not
stop ``/health`` from answering. torch, transformers and peft are imported
inside ``load_generator`` and nowhere else, so ``--fake`` runs without them.

Two real generators answer the same call. ``load_generator`` is the
transformers path (the adapter on the base model, cuda or mps), which serves a
GPU; ``load_llama_generator`` is the Mac's path, a merged GGUF behind
llama.cpp's ``llama-server`` on Metal, spoken to over its OpenAI-compatible
chat endpoint with the standard library.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

from ..data.doctrine import system_prompt, user_prompt
from ..data.prompts import ANCHOR, REGISTER, render_group
from ..reward.tasks import extract_group

TASKS = (REGISTER, ANCHOR)
DEFAULT_N = 8
# The eval's sampling temperature. Training rolled out at 1.0.
DEFAULT_TEMPERATURE = 0.8
MAX_N = 32
MAX_NEW_TOKENS = 128
TOP_P = 0.95
# A request is a group of search terms; anything near this size is not one.
MAX_BODY_BYTES = 64 * 1024


class Generator(Protocol):
    """Raw completions for one conversation: ``n`` samples at ``temperature``."""

    def __call__(
        self, messages: list[dict[str, str]], n: int, temperature: float, seed: int | None
    ) -> list[str]: ...


def build_messages(task: str, given: str) -> list[dict[str, str]]:
    """The conversation training showed the model, for one task and one given.

    The same two functions ``data.prompts.Prompt.messages`` calls, so a
    request and a training prompt with the same task and given are
    byte-identical by construction.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, not {task!r}")
    return [
        {"role": "system", "content": system_prompt(task)},
        {"role": "user", "content": user_prompt(task, given)},
    ]


# ---------------------------------------------------------------------------
# The real generator
# ---------------------------------------------------------------------------


def resolve_device(requested: str = "auto") -> str:
    """``auto`` is cuda, then mps, then cpu. Imports torch."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _ModelGenerator:
    """A loaded base model with the adapter on it, sampling under one lock."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._lock = threading.Lock()

    def __call__(
        self, messages: list[dict[str, str]], n: int, temperature: float, seed: int | None
    ) -> list[str]:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        width = encoded["input_ids"].shape[1]
        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            with torch.inference_mode():
                produced = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    top_p=TOP_P,
                    num_return_sequences=n,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        return self.tokenizer.batch_decode(produced[:, width:], skip_special_tokens=True)


def load_generator(model_id: str, adapter_path: str, device: str = "auto") -> _ModelGenerator:
    """Load the base model, put the adapter on it, and return the sampler.

    fp16 on cuda and mps, fp32 on cpu. The tokenizer is the base model's, as
    in ``eval.generate``, left-padded with eos standing in for a missing pad.
    The returned callable carries the resolved ``device``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    dtype = torch.float32 if resolved == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(resolved)
    model = PeftModel.from_pretrained(base, adapter_path).eval()
    return _ModelGenerator(model, tokenizer, resolved)


# ---------------------------------------------------------------------------
# The llama.cpp generator, for the Mac
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_PORT = 8790
# Slots in llama-server: eight samples decode side by side, one prompt pass
# shared among them.
LLAMA_PARALLEL = 8
LLAMA_HEALTH_TIMEOUT = 120.0
LLAMA_REQUEST_TIMEOUT = 300.0
# The transformers path samples through ``model.generate``, which inherits
# Qwen2.5's generation_config.json (top_k 20, repetition_penalty 1.05) under
# the temperature and top_p it passes; the eval that picked this adapter
# sampled the same way. llama-server's own defaults are top_k 40, min_p 0.05
# and no repeat penalty, so each request names the transformers values.
LLAMA_SAMPLING = {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.05}


class LlamaServerError(RuntimeError):
    """llama-server did not start, or answered a completion with an error."""


def _http_json(url: str, body: dict | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    """GET (no body) or POST JSON; the status and the decoded body, errors included."""
    import urllib.error
    import urllib.request

    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, raw.decode("utf-8", "replace")


def _log_tail(path: str | None, lines: int = 20) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return ""


class _LlamaGenerator:
    """Samples from a llama-server over its OpenAI-compatible chat endpoint.

    The messages go over as given and llama-server renders them with the
    GGUF's own chat template, the one ``apply_chat_template`` used in
    training. ``n`` is asked for in one request, which llama-server serves by
    processing the prompt once and copying it into ``n - 1`` further slots; a
    request is capped at the slot count (llama-server waits forever for more
    free slots than it has), so a larger ``n`` goes as several such requests
    side by side. Should a server return fewer choices than asked, the rest
    are made up by single-sample requests in threads. With a seed, each
    sample's seed is the request's plus its index, as llama-server itself does
    for the copies.
    """

    device = "metal"

    def __init__(
        self,
        base_url: str,
        *,
        parallel: int = LLAMA_PARALLEL,
        process=None,
        log_path: str | None = None,
        request_timeout: float = LLAMA_REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.parallel = max(1, parallel)
        self.process = process
        self.log_path = log_path
        self.request_timeout = request_timeout
        self._closed = False

    def _payload(self, messages, n: int, temperature: float, seed: int | None) -> dict:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "top_p": TOP_P,
            "max_tokens": MAX_NEW_TOKENS,
            "n": n,
            **LLAMA_SAMPLING,
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    def _complete(self, messages, n: int, temperature: float, seed: int | None) -> list[str]:
        """One chat completion request; its choices' contents in index order."""
        status, body = _http_json(
            self.base_url + "/v1/chat/completions",
            self._payload(messages, n, temperature, seed),
            timeout=self.request_timeout,
        )
        if status != 200 or not isinstance(body, dict) or "choices" not in body:
            detail = body.get("error", body) if isinstance(body, dict) else body
            raise LlamaServerError(f"llama-server answered {status}: {detail}")
        choices = sorted(body["choices"], key=lambda choice: choice.get("index", 0))
        return [(choice.get("message") or {}).get("content") or "" for choice in choices[:n]]

    def _fan_out(self, calls: list[Callable[[], list[str]]]) -> list[str]:
        """Run the calls side by side; their results concatenated in call order."""
        if len(calls) == 1:
            return calls[0]()
        results: list[list[str] | None] = [None] * len(calls)
        errors: list[BaseException] = []

        def run(index: int) -> None:
            try:
                results[index] = calls[index]()
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(calls))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return [text for result in results for text in (result or [])]

    def __call__(
        self, messages: list[dict[str, str]], n: int, temperature: float, seed: int | None
    ) -> list[str]:
        def chunk(start: int, size: int) -> Callable[[], list[str]]:
            chunk_seed = None if seed is None else seed + start
            return lambda: self._complete(messages, size, temperature, chunk_seed)

        starts = range(0, n, self.parallel)
        texts = self._fan_out([chunk(s, min(self.parallel, n - s)) for s in starts])
        if len(texts) < n:
            texts += self._fan_out([chunk(i, 1) for i in range(len(texts), n)])
        return texts[:n]

    def close(self) -> None:
        """Stop the llama-server this generator started, and drop its log; a no-op otherwise."""
        if self._closed:
            return
        self._closed = True
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001 — a hung server is killed
                process.kill()
                process.wait(timeout=10)
        if self.log_path:
            try:
                os.remove(self.log_path)
            except OSError:
                pass


def llama_server_command(
    gguf_path: str, *, port: int, n_ctx: int, parallel: int, binary: str = "llama-server"
) -> list[str]:
    """The llama-server command line. ``n_ctx`` is per slot.

    llama-server's ``-c`` is the whole KV pool, divided among the slots, so it
    is ``n_ctx`` times the slot count; every layer goes to Metal.
    """
    return [
        binary,
        "-m", gguf_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-c", str(n_ctx * parallel),
        "-ngl", "99",
        "--parallel", str(parallel),
    ]


def wait_for_health(
    base_url: str,
    *,
    timeout: float = LLAMA_HEALTH_TIMEOUT,
    process=None,
    log_path: str | None = None,
    interval: float = 0.25,
) -> None:
    """Block until ``GET /health`` says ok; raise ``LlamaServerError`` otherwise.

    llama-server answers 503 while the model loads. A spawned process that
    exits first, or a server still not ready at ``timeout``, fails loudly with
    the tail of its log.
    """
    import time

    deadline = time.monotonic() + timeout
    last = "no answer"
    while True:
        if process is not None and process.poll() is not None:
            raise LlamaServerError(
                f"llama-server exited with {process.returncode} before it was ready"
                + (f"; log {log_path}:\n{_log_tail(log_path)}" if log_path else "")
            )
        try:
            status, body = _http_json(base_url + "/health", timeout=2.0)
        except OSError as exc:
            last = str(exc)
        else:
            if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                return
            last = f"{status} {body}"
        if time.monotonic() >= deadline:
            raise LlamaServerError(
                f"llama-server at {base_url} not ready after {timeout:.0f} s ({last})"
                + (f"; log {log_path}:\n{_log_tail(log_path)}" if log_path else "")
            )
        time.sleep(interval)


def _port_answers(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def load_llama_generator(
    gguf_path: str,
    *,
    port: int = DEFAULT_LLAMA_PORT,
    n_ctx: int = 2048,
    spawn: bool = True,
    parallel: int = LLAMA_PARALLEL,
    health_timeout: float = LLAMA_HEALTH_TIMEOUT,
    binary: str = "llama-server",
) -> _LlamaGenerator:
    """A generator backed by llama-server on 127.0.0.1:``port``.

    With ``spawn`` it starts ``llama-server`` on the GGUF (the port must be
    free, so the health check cannot be answered by some other server), logs
    to a temporary file, and waits for ``/health``; ``.close()`` stops it.
    Without ``spawn`` it waits for a server someone else started on that port.
    """
    import atexit
    import shutil
    import subprocess
    import tempfile

    base_url = f"http://127.0.0.1:{port}"
    if not spawn:
        wait_for_health(base_url, timeout=health_timeout)
        return _LlamaGenerator(base_url, parallel=parallel)

    if not shutil.which(binary):
        raise LlamaServerError(f"{binary} is not on PATH (brew install llama.cpp)")
    if not os.path.isfile(gguf_path):
        raise LlamaServerError(f"no GGUF at {gguf_path}")
    if _port_answers(port):
        raise LlamaServerError(f"port {port} is already in use; choose another --llama-port")
    fd, log_path = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
    with os.fdopen(fd, "wb") as log:
        process = subprocess.Popen(
            llama_server_command(gguf_path, port=port, n_ctx=n_ctx, parallel=parallel, binary=binary),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    generator = _LlamaGenerator(base_url, parallel=parallel, process=process, log_path=log_path)
    atexit.register(generator.close)
    try:
        wait_for_health(base_url, timeout=health_timeout, process=process, log_path=log_path)
    except BaseException:
        generator.close()
        raise
    return generator


# ---------------------------------------------------------------------------
# The fake generator, for tests and for Cloris's smoke test
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r'"([^"]+)"')

# Class vocabulary for the anchor task: the doctrine's own example answer, and
# a title. Legal groups the scorer can read, which is all a smoke test needs.
_FAKE_ANCHORS = (
    '("benchmark" OR "benchmarks" OR "benchmarking" OR "evaluation" OR "evals")',
    'An anchor for this sweep would be the evaluation class of terms.',
    '("research engineer" OR "research engineers")',
    '("evaluation" AND "benchmark")',
)


def _given_of(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Recover the task and the given from a ``build_messages`` conversation."""
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    for task, head, tail in (
        (REGISTER, "Term: ", "\nOR group:"),
        (ANCHOR, "Sweep: ", "\nAnchor group:"),
    ):
        if user.startswith(head) and user.endswith(tail):
            return task, user[len(head) : -len(tail)]
    return REGISTER, user


def _register_forms(given: str) -> list[str]:
    terms = _QUOTED_RE.findall(given) or [given.strip("()\" \n")]
    forms: list[str] = []
    for term in terms:
        forms.append(term)
        if not term.lower().endswith("s"):
            forms.append(term + "s")
        if "-" in term:
            forms.append(term.replace("-", " "))
        elif " " in term:
            forms.append(term.replace(" ", "-"))
    return [form for form in forms if form.strip()]


def fake_generator(
    messages: list[dict[str, str]], n: int, temperature: float, seed: int | None = None
) -> list[str]:
    """Canned completions, deterministic, ignoring temperature and seed.

    The cycle always starts with a legal single OR group and always holds a
    prose answer, so a caller sees both a non-null and a null ``group`` from
    any ``n`` of two or more. Register answers are built from the given's own
    terms (plurals and hyphen twins), so a candidate lands on the concept the
    caller asked about.
    """
    task, given = _given_of(messages)
    if task == ANCHOR:
        pool = list(_FAKE_ANCHORS)
    else:
        forms = _register_forms(given)
        expanded = render_group(forms[:16])
        pool = [
            expanded,
            f"A good sourcer would write {render_group(forms[:1])} with its plurals.",
            "```\n" + render_group(forms[:2]) + "\n```",
            f"{render_group(forms[:1])} AND {expanded}",
        ]
    return [pool[index % len(pool)] for index in range(n)]


# ---------------------------------------------------------------------------
# The request and the handler
# ---------------------------------------------------------------------------


class BadRequest(ValueError):
    """A request the contract refuses; the message is the 400's detail."""


@dataclass(frozen=True)
class EnrichRequest:
    task: str
    given: str
    n: int = DEFAULT_N
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_request(payload: Any) -> EnrichRequest:
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")
    task = payload.get("task")
    if task not in TASKS:
        raise BadRequest(f"task must be one of {list(TASKS)}, not {task!r}")
    given = payload.get("given")
    if not isinstance(given, str) or not given.strip():
        raise BadRequest("given must be a non-empty string: the group text as authored")
    n = payload.get("n", DEFAULT_N)
    if not _is_int(n) or not 1 <= n <= MAX_N:
        raise BadRequest(f"n must be an integer from 1 to {MAX_N}, not {n!r}")
    temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise BadRequest(f"temperature must be a positive number, not {temperature!r}")
    seed = payload.get("seed")
    if seed is not None and not _is_int(seed):
        raise BadRequest(f"seed must be an integer or null, not {seed!r}")
    return EnrichRequest(task, given, n, float(temperature), seed)


def make_handler(
    generator: Generator | Callable[..., list[str]],
    *,
    model_id: str,
    adapter_label: str,
    device: str,
    verbose: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """A request handler class bound to one generator and its labels."""

    class SidecarHandler(BaseHTTPRequestHandler):
        server_version = "cloris-boolean-sidecar/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            if verbose:
                super().log_message(format, *args)

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _not_found(self) -> None:
            self._send(404, {"detail": f"no route for {self.command} {urlsplit(self.path).path}"})

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/health":
                return self._not_found()
            self._send(
                200, {"ok": True, "model": model_id, "adapter": adapter_label, "device": device}
            )

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/enrich":
                return self._not_found()
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._send(400, {"detail": "Content-Length is not an integer"})
            if length > MAX_BODY_BYTES:
                return self._send(413, {"detail": f"body over {MAX_BODY_BYTES} bytes"})
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return self._send(400, {"detail": f"the body is not JSON: {exc}"})
            try:
                request = parse_request(payload)
            except BadRequest as exc:
                return self._send(400, {"detail": str(exc)})
            messages = build_messages(request.task, request.given)
            try:
                completions = generator(messages, request.n, request.temperature, request.seed)
                candidates = [
                    {"text": text, "group": extract_group(text)} for text in completions
                ]
            except Exception as exc:  # noqa: BLE001 — any failure is the caller's 500
                return self._send(500, {"detail": str(exc) or type(exc).__name__})
            self._send(200, {"adapter": adapter_label, "candidates": candidates})

        do_PUT = do_DELETE = do_PATCH = _not_found  # noqa: N815

    return SidecarHandler


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def make_server(
    host: str, port: int, handler_cls: type[BaseHTTPRequestHandler]
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    return server


def serve(
    host: str,
    port: int,
    handler_cls: type[BaseHTTPRequestHandler],
    *,
    on_ready: Callable[[str], None] | None = None,
) -> None:
    """Serve until Ctrl-C, then close the socket and return."""
    server = make_server(host, port, handler_cls)
    bound_host, bound_port = server.server_address[:2]
    if on_ready is not None:
        on_ready(f"http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
