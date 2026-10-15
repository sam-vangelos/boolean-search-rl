"""Renew the colab CLI's runtime-proxy token for a session, or adopt an orphan.

google-colab-cli 0.6.0 stores a proxy token that expires after 3,600 s and never
renews it; after that every command gets 404 and the CLI prunes the session while
the VM keeps running (upstream issue #106). ``list_assignments()`` returns a fresh
token for every live assignment, so run this before every ``colab`` call:

    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_refresh.py NAME
    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_refresh.py NAME --adopt ENDPOINT
    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_refresh.py NAME --adopt auto

``--adopt`` re-registers a pruned session (and restarts its keep-alive daemon);
``auto`` picks the assignment when exactly one is live. Exits non-zero when the
endpoint is not assigned. From plans/research/colab-cli.md (2026-09-22).
"""
import sys

from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.commands.session import spawn_keep_alive

name = sys.argv[1]
adopt = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--adopt" else None
s = state.store.get(name)
assignments = state.client.list_assignments()
if adopt == "auto":
    owned = {x.endpoint for x in state.store.list().values()} if hasattr(state.store, "list") else set()
    orphans = [a for a in assignments if a.endpoint not in owned]
    if len(orphans) != 1:
        sys.exit(f"cannot auto-adopt: {len(orphans)} unowned assignments of {len(assignments)} live")
    adopt = orphans[0].endpoint
endpoint = adopt or (s.endpoint if s else None)
a = next((a for a in assignments if a.endpoint == endpoint), None)
if a is None:
    sys.exit(f"{endpoint} is not assigned")
if s is None or adopt:
    s = SessionState(
        name=name, token="", url="", endpoint=a.endpoint,
        variant=a.variant.name, accelerator=a.accelerator.value,
    )
    state.store.add(s)
    s.keep_alive_pid = spawn_keep_alive(
        a.endpoint, name, auth_provider=state.auth_provider, config_path=state.config_path
    )
    print(f"adopted {a.endpoint} as {name}; keep-alive pid {s.keep_alive_pid}")
s.token, s.url = a.runtime_proxy_info.token, a.runtime_proxy_info.url
state.store.add(s)
print(f"refreshed token for {name} ({a.endpoint})")
