"""The sidecar: its contract, driven over loopback with a fake generator.

No model loads here and nothing leaves 127.0.0.1. The handler is served on an
ephemeral port in a thread and spoken to with urllib, the same client Cloris
uses, so what these tests pin is the wire contract rather than the handler's
internals.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cloris_boolean_rl.data.prompts import ANCHOR, REGISTER, Prompt, load_prompts
from cloris_boolean_rl.reward.tasks import extract_group
from cloris_boolean_rl.serve import sidecar

ROOT = Path(__file__).resolve().parents[1]


def _start(generator) -> tuple[str, object]:
    handler = sidecar.make_handler(
        generator, model_id="test-model", adapter_label="test-adapter", device="cpu"
    )
    server = sidecar.make_server("127.0.0.1", 0, handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}", server


@pytest.fixture(scope="module")
def fake_url():
    url, server = _start(sidecar.fake_generator)
    yield url
    server.shutdown()
    server.server_close()


def _call(url: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(url + path, data=body, method="POST" if body is not None else "GET")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def _enrich(url: str, payload) -> tuple[int, dict]:
    return _call(url, "/enrich", json.dumps(payload).encode())


# ---------------------------------------------------------------------------
# The two endpoints
# ---------------------------------------------------------------------------


def test_health_reports_model_adapter_and_device(fake_url):
    status, body = _call(fake_url, "/health")
    assert status == 200
    assert body == {"ok": True, "model": "test-model", "adapter": "test-adapter", "device": "cpu"}


def test_a_register_request_returns_n_candidates_with_groups_extracted(fake_url):
    status, body = _enrich(fake_url, {"task": "register", "given": '"evaluation"', "n": 4})
    assert status == 200
    assert body["adapter"] == "test-adapter"
    candidates = body["candidates"]
    assert len(candidates) == 4
    for candidate in candidates:
        assert set(candidate) == {"text", "group"}
        # The group is exactly what the reward's own contract check says.
        assert candidate["group"] == extract_group(candidate["text"])
    groups = [c["group"] for c in candidates if c["group"] is not None]
    assert groups, "at least one candidate keeps the contract"
    assert any(c["group"] is None for c in candidates), "a prose completion gives null"
    assert all(extract_group(group) == group for group in groups)


def test_the_default_n_is_eight(fake_url):
    status, body = _enrich(fake_url, {"task": "register", "given": '"co-founder"'})
    assert status == 200
    assert len(body["candidates"]) == 8


def test_an_anchor_request_works(fake_url):
    sweep = '("IFEval" OR "RewardBench" OR "MT-Bench")'
    status, body = _enrich(fake_url, {"task": "anchor", "given": sweep, "n": 3})
    assert status == 200
    assert len(body["candidates"]) == 3
    assert any(c["group"] is not None for c in body["candidates"])


# ---------------------------------------------------------------------------
# Errors are JSON with a detail
# ---------------------------------------------------------------------------


def test_bad_json_is_a_400_with_detail(fake_url):
    status, body = _call(fake_url, "/enrich", b"{not json")
    assert status == 400
    assert "detail" in body and body["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"task": "summarise", "given": '"evaluation"'},
        {"given": '"evaluation"'},
        {"task": "register"},
        {"task": "register", "given": 7},
        {"task": "register", "given": '"evaluation"', "n": 0},
        {"task": "register", "given": '"evaluation"', "n": 33},
        {"task": "register", "given": '"evaluation"', "n": "8"},
        {"task": "register", "given": '"evaluation"', "n": True},
        {"task": "register", "given": '"evaluation"', "temperature": "hot"},
        {"task": "register", "given": '"evaluation"', "temperature": 0},
        {"task": "register", "given": '"evaluation"', "seed": "one"},
        ["register", '"evaluation"'],
    ],
)
def test_an_invalid_request_is_a_400_with_detail(fake_url, payload):
    status, body = _enrich(fake_url, payload)
    assert status == 400
    assert body.get("detail")


def test_an_unknown_path_is_a_json_404(fake_url):
    status, body = _call(fake_url, "/nope")
    assert status == 404
    assert body.get("detail")
    status, body = _call(fake_url, "/nope", b"{}")
    assert status == 404
    assert body.get("detail")


def test_a_generator_exception_is_a_500_with_detail():
    def broken(messages, n, temperature, seed=None):
        raise RuntimeError("MPS backend out of memory")

    url, server = _start(broken)
    try:
        status, body = _enrich(url, {"task": "register", "given": '"evaluation"', "n": 2})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 500
    assert "out of memory" in body["detail"]


def test_the_generator_receives_the_request_it_was_sent():
    seen = {}

    def recording(messages, n, temperature, seed=None):
        seen.update(messages=messages, n=n, temperature=temperature, seed=seed)
        return ['("evaluation" OR "evals")'] * n

    url, server = _start(recording)
    try:
        status, _ = _enrich(
            url, {"task": "anchor", "given": '("IFEval")', "n": 2, "temperature": 1.0, "seed": 11}
        )
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    assert seen["messages"] == sidecar.build_messages("anchor", '("IFEval")')
    assert (seen["n"], seen["temperature"], seen["seed"]) == (2, 1.0, 11)


# ---------------------------------------------------------------------------
# The model sees what training showed it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", [REGISTER, ANCHOR])
def test_messages_equal_the_training_prompts(task):
    prompt = next(p for p in load_prompts() if p.task == task)
    assert sidecar.build_messages(prompt.task, prompt.given) == prompt.messages
    rebuilt = Prompt(id="x", task=task, split="eval", source="t", given=prompt.given, targets=())
    assert sidecar.build_messages(task, prompt.given) == rebuilt.messages


def test_build_messages_refuses_an_unknown_task():
    with pytest.raises(ValueError):
        sidecar.build_messages("summarise", '"evaluation"')


def test_the_fake_generator_is_deterministic():
    messages = sidecar.build_messages("register", '"co-founder"')
    first = sidecar.fake_generator(messages, 6, 0.8, None)
    assert first == sidecar.fake_generator(messages, 6, 1.0, 5)
    assert len(first) == 6


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_the_fake_cli_never_imports_torch():
    script = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('serve_sidecar', {str(ROOT / 'tools' / 'serve_sidecar.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "args = module.parse_args(['--fake'])\n"
        "handler, info = module.build(args)\n"
        "assert info['adapter'] == 'fake', info\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers was imported'\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_the_adapter_label_is_the_release_directory():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "serve_sidecar_labels", ROOT / "tools" / "serve_sidecar.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.adapter_label("release/re-v3-3b-ck100-20260923-0303/adapter")
        == "re-v3-3b-ck100-20260923-0303"
    )
    assert module.adapter_label("release/re-v3-3b-ck100-20260923-0303/adapter/") == (
        "re-v3-3b-ck100-20260923-0303"
    )
    assert module.adapter_label("runs/re-v3/checkpoint-100") == "checkpoint-100"


# ---------------------------------------------------------------------------
# The llama.cpp generator, against a fake llama-server
# ---------------------------------------------------------------------------


class _FakeLlama:
    """Mimics llama-server's ``/health`` and ``/v1/chat/completions``.

    Records every completion request. ``honour_n`` False answers one choice
    whatever ``n`` asks for, as an older server would; ``not_ready`` is how
    many health checks answer 503 (loading) before the first ok; ``fail``
    makes every completion a 500.
    """

    def __init__(self, *, honour_n: bool = True, not_ready: int = 0, fail: bool = False):
        self.requests: list[dict] = []
        self.health_checks = 0
        self.honour_n, self.not_ready, self.fail = honour_n, not_ready, fail
        self._lock = threading.Lock()
        fake = self

        class Handler(sidecar.BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                with fake._lock:
                    fake.health_checks += 1
                    loading = fake.health_checks <= fake.not_ready
                if loading:
                    return self._send(503, {"error": {"code": 503, "message": "Loading model"}})
                self._send(200, {"status": "ok"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake._lock:
                    fake.requests.append(body)
                    ordinal = len(fake.requests)
                if fake.fail:
                    return self._send(500, {"error": {"code": 500, "message": "slot crashed"}})
                count = body.get("n", 1) if fake.honour_n else 1
                # Choices come back out of order; the generator sorts by index.
                choices = [
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": f'("r{ordinal}" OR "c{i}")'},
                        "finish_reason": "stop",
                    }
                    for i in reversed(range(count))
                ]
                self._send(200, {"object": "chat.completion", "choices": choices})

        self.server = sidecar.make_server("127.0.0.1", 0, Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_llama():
    made: list[_FakeLlama] = []

    def make(**kwargs) -> _FakeLlama:
        made.append(_FakeLlama(**kwargs))
        return made[-1]

    yield make
    for fake in made:
        fake.close()


def test_the_llama_generator_posts_the_sampling_contract(fake_llama):
    fake = fake_llama()
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False)
    messages = sidecar.build_messages("register", '"evaluation"')
    texts = generator(messages, 8, 0.8, 7)
    assert len(fake.requests) == 1, "eight samples are one request, one prompt pass"
    request = fake.requests[0]
    assert request["messages"] == messages
    assert request["temperature"] == 0.8
    assert request["top_p"] == sidecar.TOP_P == 0.95
    assert request["max_tokens"] == sidecar.MAX_NEW_TOKENS == 128
    assert request["n"] == 8
    assert request["seed"] == 7
    # The transformers path's inherited sampler, named explicitly.
    assert (request["top_k"], request["min_p"], request["repeat_penalty"]) == (20, 0.0, 1.05)
    assert texts == [f'("r1" OR "c{i}")' for i in range(8)]
    assert generator.device == "metal"
    generator.close()  # nothing spawned, so nothing to stop


def test_the_llama_generator_sends_no_seed_when_none_is_given(fake_llama):
    fake = fake_llama()
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False)
    generator(sidecar.build_messages("anchor", '("IFEval")'), 2, 1.0, None)
    assert "seed" not in fake.requests[0]
    assert fake.requests[0]["temperature"] == 1.0


def test_n_above_the_slot_count_is_split_into_slot_sized_requests(fake_llama):
    fake = fake_llama()
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False, parallel=8)
    texts = generator(sidecar.build_messages("register", '"evaluation"'), 11, 0.8, 100)
    assert len(texts) == 11
    assert sorted(r["n"] for r in fake.requests) == [3, 8]
    # Each sample's seed is the request's plus its index, as llama-server does
    # for the copies inside one request.
    assert sorted(r["seed"] for r in fake.requests) == [100, 108]


def test_a_server_that_ignores_n_is_made_up_with_single_requests(fake_llama):
    fake = fake_llama(honour_n=False)
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False)
    texts = generator(sidecar.build_messages("register", '"evaluation"'), 4, 0.8, 5)
    assert len(texts) == 4
    assert [r["n"] for r in fake.requests[:1]] == [4]
    assert sorted(r["n"] for r in fake.requests[1:]) == [1, 1, 1]
    assert sorted(r["seed"] for r in fake.requests[1:]) == [6, 7, 8]


def test_no_spawn_waits_for_a_loading_server(fake_llama):
    fake = fake_llama(not_ready=3)
    sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False, health_timeout=10)
    assert fake.health_checks == 4


def test_a_server_that_never_gets_ready_fails_loudly(fake_llama):
    fake = fake_llama(not_ready=10_000)
    with pytest.raises(sidecar.LlamaServerError, match="not ready"):
        sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False, health_timeout=0.5)


def test_the_llama_generator_serves_the_sidecar_contract(fake_llama):
    fake = fake_llama()
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False)
    handler = sidecar.make_handler(
        generator, model_id="m", adapter_label="a", device=generator.device
    )
    server = sidecar.make_server("127.0.0.1", 0, handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, body = _enrich(url, {"task": "register", "given": '"evaluation"', "n": 8})
        health = _call(url, "/health")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    assert len(body["candidates"]) == 8
    assert all(c["group"] == extract_group(c["text"]) is not None for c in body["candidates"])
    assert health == (200, {"ok": True, "model": "m", "adapter": "a", "device": "metal"})


def test_a_llama_server_error_is_the_sidecars_500(fake_llama):
    fake = fake_llama(fail=True)
    generator = sidecar.load_llama_generator("unused.gguf", port=fake.port, spawn=False)
    url, server = _start(generator)
    try:
        status, body = _enrich(url, {"task": "register", "given": '"evaluation"', "n": 2})
    finally:
        server.shutdown()
        server.server_close()
    assert status == 500
    assert "slot crashed" in body["detail"]


def test_the_llama_server_command_sizes_the_context_per_slot():
    command = sidecar.llama_server_command("/m/model-Q8_0.gguf", port=8790, n_ctx=2048, parallel=8)
    assert command[0] == "llama-server"
    pairs = dict(zip(command[1::2], command[2::2]))
    assert pairs == {
        "-m": "/m/model-Q8_0.gguf",
        "--port": "8790",
        "--host": "127.0.0.1",
        "-c": str(2048 * 8),
        "-ngl": "99",
        "--parallel": "8",
    }


_FAKE_LLAMA_SERVER = '''#!{python}
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        data = json.dumps({{"status": "ok"}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_spawn_starts_the_binary_waits_for_health_and_close_stops_it(tmp_path):
    binary = tmp_path / "fake-llama-server"
    binary.write_text(_FAKE_LLAMA_SERVER.format(python=sys.executable))
    binary.chmod(0o755)
    gguf = tmp_path / "model-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    generator = sidecar.load_llama_generator(
        str(gguf), port=_free_port(), binary=str(binary), health_timeout=30
    )
    process = generator.process
    log_path = generator.log_path
    try:
        assert process.poll() is None
        assert process.args[:3] == [str(binary), "-m", str(gguf)]
        assert Path(log_path).is_file()
    finally:
        generator.close()
    assert process.poll() is not None, "close() stopped the server"
    assert not Path(log_path).exists(), "close() dropped the log"
    generator.close()  # twice is harmless


def test_spawn_refuses_a_port_already_in_use(fake_llama, tmp_path):
    fake = fake_llama()
    gguf = tmp_path / "model-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(sidecar.LlamaServerError, match="already in use"):
        sidecar.load_llama_generator(str(gguf), port=fake.port, binary=sys.executable)


def test_spawn_fails_loudly_without_the_binary(tmp_path):
    with pytest.raises(sidecar.LlamaServerError, match="not on PATH"):
        sidecar.load_llama_generator(
            str(tmp_path / "m.gguf"), port=_free_port(), binary="no-such-llama-server-binary"
        )


def test_a_spawned_server_that_exits_fails_loudly(tmp_path):
    binary = tmp_path / "dying-llama-server"
    binary.write_text(f"#!{sys.executable}\nimport sys\nprint('failed to load model')\nsys.exit(3)\n")
    binary.chmod(0o755)
    gguf = tmp_path / "model-Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(sidecar.LlamaServerError, match="exited with 3") as caught:
        sidecar.load_llama_generator(str(gguf), port=_free_port(), binary=str(binary))
    assert "failed to load model" in str(caught.value)


# ---------------------------------------------------------------------------
# The CLI's --gguf
# ---------------------------------------------------------------------------


def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("serve_sidecar_gguf", ROOT / "tools" / "serve_sidecar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gguf_label_is_the_release_directory():
    cli = _cli()
    assert (
        cli.gguf_label("release/re-v3-3b-ck100-20260923-0303/gguf/model-Q8_0.gguf")
        == "re-v3-3b-ck100-20260923-0303"
    )
    assert cli.gguf_label("/tmp/elsewhere/model-Q8_0.gguf") == "model-Q8_0"


def test_gguf_and_adapter_are_mutually_exclusive(capsys):
    cli = _cli()
    with pytest.raises(SystemExit):
        cli.parse_args(["--gguf", "m.gguf", "--adapter", "a"])
    args = cli.parse_args(["--gguf", "m.gguf"])
    assert args.gguf == "m.gguf" and args.adapter is None
    assert args.llama_port == 8790
    assert cli.parse_args(["--gguf", "m.gguf", "--llama-port", "9001"]).llama_port == 9001


def test_gguf_selects_the_llama_backend(monkeypatch):
    cli = _cli()
    calls = []

    class Stub:
        device = "metal"

        def __call__(self, messages, n, temperature, seed):
            return ['("x")'] * n

        def close(self):
            calls.append("close")

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return Stub()

    monkeypatch.setattr(cli.sidecar, "load_llama_generator", fake_load)
    path = "release/re-v3-3b-ck100-20260923-0303/gguf/model-Q8_0.gguf"
    generator, info = cli.build_generator(cli.parse_args(["--gguf", path, "--llama-port", "8791"]))
    assert calls == [(path, {"port": 8791})]
    assert info == {
        "model": cli.DEFAULT_MODEL,
        "adapter": "re-v3-3b-ck100-20260923-0303",
        "device": "metal",
    }
    generator.close()
    assert calls[-1] == "close"
