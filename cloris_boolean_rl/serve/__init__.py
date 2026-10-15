"""The sidecar: the trained adapter behind a small local HTTP service.

Cloris asks it for candidate groups and scores them itself. Importing this
package never imports torch; only ``sidecar.load_generator`` does, and only
when it is called. ``sidecar.load_llama_generator`` is the Mac's backend, a
merged GGUF behind llama-server, and needs nothing beyond the standard library.
"""

from .sidecar import (
    DEFAULT_LLAMA_PORT,
    DEFAULT_N,
    DEFAULT_TEMPERATURE,
    TASKS,
    Generator,
    LlamaServerError,
    build_messages,
    fake_generator,
    load_generator,
    load_llama_generator,
    make_handler,
    make_server,
    serve,
)

__all__ = [
    "DEFAULT_LLAMA_PORT",
    "DEFAULT_N",
    "DEFAULT_TEMPERATURE",
    "TASKS",
    "Generator",
    "LlamaServerError",
    "build_messages",
    "fake_generator",
    "load_generator",
    "load_llama_generator",
    "make_handler",
    "make_server",
    "serve",
]
